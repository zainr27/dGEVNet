# dGEVNet

A PyTorch implementation of a deep Generalized Extreme Value (GEV) network for rainfall modeling.

## Description

dGEVNet is a neural network model that learns the parameters of a duration-dependent GEV distribution for rainfall modeling. The model takes location-specific covariates and time-varying covariates as input to predict GEV parameters.

## Features

- Duration-dependent GEV parameter estimation
- Spatial smoothness regularization
- Stable numerical implementation
- Support for multiple location and time-varying covariates

## Requirements

- Python 3.7+
- PyTorch
- NumPy

## Installation

```bash
pip install torch numpy
```

## Usage

```python
from dgevnet import dGEVNet

# Initialize model
model = dGEVNet(M=3, P=2)  # M: location covariates, P: time-varying covariates

# Train model
# See example usage in the code
```

## License

MIT License 