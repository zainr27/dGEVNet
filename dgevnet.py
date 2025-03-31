import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dGEVNet(nn.Module):
    def __init__(self, M: int, P: int, hidden_size: int = 64, dropout_rate: float = 0.3):
        """
        Args:
            M (int): Number of location-specific covariates.
            P (int): Number of time-varying covariates.
            hidden_size (int): Number of neurons in hidden layers.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(dGEVNet, self).__init__()
        self.M = M
        self.P = P
        self.output_dim = 8 + 3 * P  # 8 base params + 3 coefficients per time covariate
        
        self.network = nn.Sequential(
            nn.Linear(M, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, self.output_dim)
        )
        
        # Initialize weights for stability
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (Tensor): Location covariates [batch_size, M]
        Returns:
            Tensor: dGEV parameters [batch_size, 8 + 3P]
        """
        return self.network(z)
    
    def compute_gev_params(self, outputs: torch.Tensor, x: torch.Tensor, d: torch.Tensor, P: int) -> tuple:
        """
        Args:
            outputs (Tensor): Neural network outputs [batch_size, 8 + 3P]
            x (Tensor): Time-varying covariates [batch_size, P]
            d (Tensor): Durations [batch_size, 1]
            P (int): Number of time-varying covariates
        Returns:
            tuple: (mu, sigma, xi) GEV parameters [batch_size, 1]
        """
        # Extract and clamp base parameters for stability
        log_theta = torch.clamp(outputs[:, 0:1], min=-10, max=10)
        log_tau1 = torch.clamp(outputs[:, 1:2], min=-10, max=10)
        log_tau2 = torch.clamp(outputs[:, 2:3], min=-10, max=10)
        eta2 = torch.clamp(outputs[:, 3:4], min=-5, max=5)
        alpha_mu0 = torch.clamp(outputs[:, 4:5], min=-10, max=10)
        alpha_log_sigma0 = torch.clamp(outputs[:, 5:6], min=-10, max=10)
        alpha_eta1 = torch.clamp(outputs[:, 6:7], min=-5, max=5)
        log_xi0 = torch.clamp(outputs[:, 7:8], min=-5, max=1)  # xi should be small for stability

        # Extract and clamp beta coefficients
        beta_start = 8
        beta_mu0 = torch.clamp(outputs[:, beta_start:beta_start + P], min=-5, max=5)
        beta_log_sigma0 = torch.clamp(outputs[:, beta_start + P:beta_start + 2*P], min=-5, max=5)
        beta_eta1 = torch.clamp(outputs[:, beta_start + 2*P:beta_start + 3*P], min=-5, max=5)

        # Compute time-varying components with stability checks
        mu0 = alpha_mu0 + torch.sum(beta_mu0 * x, dim=1, keepdim=True)
        log_sigma0 = alpha_log_sigma0 + torch.sum(beta_log_sigma0 * x, dim=1, keepdim=True)
        eta1 = torch.clamp(alpha_eta1 + torch.sum(beta_eta1 * x, dim=1, keepdim=True), min=-5, max=5)

        # Compute duration adjustment with stability
        d_plus_theta = torch.clamp(d + torch.exp(log_theta), min=1e-6)

        # Compute GEV parameters with stability checks
        mu = mu0 * torch.clamp(d_plus_theta.pow(-eta1), min=1e-6, max=1e6) + torch.exp(log_tau1)
        sigma = torch.exp(log_sigma0) * torch.clamp(d_plus_theta.pow(-eta1 + eta2), min=1e-6, max=1e6) + torch.exp(log_tau2)
        xi = torch.exp(log_xi0)  # Already clamped via log_xi0

        return mu, sigma, xi
    
    def gev_nll(self, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (Tensor): Observed rainfall [batch_size, 1]
            mu (Tensor): Location parameter [batch_size, 1]
            sigma (Tensor): Scale parameter [batch_size, 1]
            xi (Tensor): Shape parameter [batch_size, 1]
        Returns:
            Tensor: Mean negative log-likelihood
        """
        z = (y - mu) / sigma
        term = 1 + xi * z
        # Clamp to ensure numerical stability
        term = torch.clamp(term, min=1e-8)
        nll = torch.log(sigma) + (1 + 1 / xi) * torch.log(term) + term.pow(-1 / xi)
        # Handle invalid values
        mask = (term > 0) & torch.isfinite(nll)
        return nll[mask].mean() if mask.any() else torch.tensor(0.0, device=device)

    def compute_laplacian(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid (Tensor): Parameter field [N_lat, N_lon]
        Returns:
            Tensor: Laplacian of the grid [N_lat, N_lon]
        """
        laplacian = torch.zeros_like(grid)
        laplacian[1:-1, 1:-1] = (
            grid[2:, 1:-1] + grid[:-2, 1:-1] + grid[1:-1, 2:] + grid[1:-1, :-2] - 4 * grid[1:-1, 1:-1]
        )
        return laplacian

    def smoothness_penalty(self, model: nn.Module, z_grid: torch.Tensor, N_lat: int, N_lon: int, P: int,
                          lambda_smooth: float = 0.01) -> torch.Tensor:
        """
        Args:
            model (nn.Module): The dGEVNet model
            z_grid (Tensor): Grid of location covariates [N_lat * N_lon, M]
            N_lat (int): Number of latitude points
            N_lon (int): Number of longitude points
            P (int): Number of time-varying covariates
            lambda_smooth (float): Smoothness penalty weight
        Returns:
            Tensor: Smoothness loss
        """
        grid_outputs = model(z_grid)  # [N_lat * N_lon, 8 + 3P]
        grid_params = grid_outputs.view(N_lat, N_lon, 8 + 3 * P)
        smoothness_loss = 0.0
        for p in range(8 + 3 * P):
            param_grid = grid_params[:, :, p]
            laplacian = self.compute_laplacian(param_grid)
            smoothness_loss += (laplacian ** 2).sum()
        return lambda_smooth * smoothness_loss

# Training code should be outside the class definition
if __name__ == "__main__":
    # Synthetic data parameters
    N = 1000  # Number of observations
    M = 3     # Location covariates (e.g., lat, lon, elev)
    P = 2     # Time-varying covariates

    # Generate random data
    z = torch.randn(N, M)           # Location covariates
    x = torch.randn(N, P)           # Time-varying covariates
    d = torch.abs(torch.randn(N, 1)) + 1  # Durations (positive)
    y = torch.abs(torch.randn(N, 1)) * 10  # Rainfall maxima

    # Create DataLoader
    dataset = TensorDataset(z, x, d, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define spatial grid (assuming z includes lat, lon, elev)
    N_lat, N_lon = 20, 20
    lat_values = torch.linspace(-1, 1, N_lat)  # Replace with actual ranges
    lon_values = torch.linspace(-1, 1, N_lon)
    lat_grid, lon_grid = torch.meshgrid(lat_values, lon_values, indexing='ij')
    mean_elev = z[:, 2].mean()  # Assuming elev is the third covariate
    z_grid = torch.stack([lat_grid.flatten(), lon_grid.flatten(),
                        mean_elev * torch.ones_like(lat_grid.flatten())], dim=1).to(device)

    # Initialize model and optimizer
    model = dGEVNet(M=M, P=P).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lambda_smooth = 0.01
    num_epochs = 50

    # Move data to device
    z, x, d, y = z.to(device), x.to(device), d.to(device), y.to(device)

    # Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_z, batch_x, batch_d, batch_y in train_loader:
            batch_z, batch_x, batch_d, batch_y = (batch_z.to(device), batch_x.to(device),
                                                batch_d.to(device), batch_y.to(device))
            optimizer.zero_grad()

            # Compute GEV parameters
            outputs = model(batch_z)
            mu, sigma, xi = model.compute_gev_params(outputs, batch_x, batch_d, P)    

            # Compute NLL loss
            nll_loss = model.gev_nll(batch_y, mu, sigma, xi)

            # Compute smoothness penalty
            smooth_loss = model.smoothness_penalty(model, z_grid, N_lat, N_lon, P, lambda_smooth)

            # Total loss
            loss = nll_loss + smooth_loss
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            if torch.isnan(torch.tensor(avg_loss)):
                print("NaN loss detected - check your data and parameters")