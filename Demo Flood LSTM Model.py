import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

# Custom dataset with week-based batching
class RiverDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Updated LSTM model with dropout
class RiverLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(RiverLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Custom loss function: alpha * (1 - NSE) + (1 - alpha) * RMSE
def custom_loss(output, target, alpha=0.5):
    # NSE calculation
    mean_target = torch.mean(target)
    nse = 1 - torch.sum((output - target) ** 2) / torch.sum((target - mean_target) ** 2)
    nse_loss = 1 - nse  # Convert to loss (minimize 1 - NSE)
    
    # RMSE calculation
    rmse = torch.sqrt(torch.mean((output - target) ** 2))
    
    return alpha * nse_loss + (1 - alpha) * rmse

# KGE metric for evaluation
def compute_kge(predictions, actuals):
    r = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    alpha = np.std(predictions) / np.std(actuals)
    beta = np.mean(predictions) / np.mean(actuals)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def prepare_data(file_path, sequence_length=14, train_split=0.8):  # Increased to 14 days
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    
    feature_columns = ['SF_absolute_humidity', 'SF_relative_humidity', 'SF_mean_air_temperature', 'SF_atmospheric_pressure', 'SF_potential_evaporation', 'SF_net_radiation', 'SF_volumetric_water_content', 'SF_soil_temperature', 'SF_wind_speed', 'catchment_daily_precipitation_armley', 'catchment_daily_precipitation_kildwick', 'discharge_armley', 'discharge_kildwick', 'river_level_snaygill', 'river_level_kildwick', 'river_level_kirkstall', 'headingley_precipitation', 'malham_precipitation', 'skipton_snaygill_precipitation', 'farnley_hall_precipitation', 'embsay_precipitation', 'silsden_precipitation', 'lower_laithe_precipitation']  # Your original features
    target_column = 'discharge_armley'
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(df[feature_columns])
    y = scaler_y.fit_transform(df[[target_column]])
    
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    train_size = int(len(X_sequences) * train_split)
    X_train = X_sequences[:train_size]
    X_test = X_sequences[train_size:]
    y_train = y_sequences[:train_size]
    y_test = y_sequences[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler_y, feature_columns

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch {epoch}: {loss.item()}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def main():
    torch.manual_seed(42)
    sequence_length = 14  # 2 weeks
    hidden_size = 32
    num_layers = 2  # Increased for capacity, with dropout
    num_epochs = 500  # Reduced with early stopping potential
    batch_size = 14   # Matches sequence_length for week-based batching
    learning_rate = 0.0001
    dropout_rate = 0.2
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'river_aire_discharge_timeseries.csv')
    
    X_train, X_test, y_train, y_test, scaler_y, feature_columns = prepare_data(csv_path, sequence_length)
    train_dataset = RiverDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for temporal order
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RiverLSTM(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout_rate=dropout_rate
    ).to(device)
    
    criterion = custom_loss  # Use custom loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_dataset = RiverDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        predictions, actuals = [], []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.numpy())
    
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))
    
    # Metrics
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    kge = compute_kge(predictions, actuals)
    
    print(f"\nTest Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"KGE: {kge:.4f}")
    
    # Persistence baseline (last value in sequence)
    persistence = scaler_y.inverse_transform(y_test[:-1].reshape(-1, 1))
    actuals_persist = scaler_y.inverse_transform(y_test[1:].reshape(-1, 1))
    kge_persist = compute_kge(persistence, actuals_persist)
    print(f"Persistence KGE: {kge_persist:.4f}")
    
    return predictions, actuals

if __name__ == "__main__":
    predictions, actuals = main()
    print("Training completed. Shape of predictions:", predictions.shape)