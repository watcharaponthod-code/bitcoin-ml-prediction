# Model Files

All trained models are included in this repository under the `models/` directory.

## Available Models

| File | Size | Type | Description |
|------|------|------|-------------|
| bitcoin_lstm_model.keras | 1.6 MB | LSTM | Deep learning model for price regression |
| bitcoin_xgb_model.pkl | 1.1 MB | XGBoost | Gradient boosting classifier |
| bitcoin_rf_model.pkl | 5.4 MB | Random Forest | Ensemble classifier |
| scaler_features.pkl | 959 B | Scaler | MinMaxScaler for features |
| scaler_close.pkl | 719 B | Scaler | MinMaxScaler for price |

## Usage

### Load Models from Repository

```python
import joblib
from tensorflow.keras.models import load_model

# Clone the repository first
# git clone https://github.com/watcharaponthod-code/bitcoin-ml-prediction.git

# Load models
lstm_model = load_model('models/bitcoin_lstm_model.keras')
xgb_model = joblib.load('models/bitcoin_xgb_model.pkl')
rf_model = joblib.load('models/bitcoin_rf_model.pkl')

# Load scalers
scaler_features = joblib.load('models/scaler_features.pkl')
scaler_close = joblib.load('models/scaler_close.pkl')
```

### Alternative: Download from Google Drive

Models are also available on Google Drive: [Bitcoin-ML-Models](https://drive.google.com/drive/folders/1qpKcXxEf4aPNdUxSPRWnJ_0kVbXAlBt1)

```python
import gdown

# LSTM Model
gdown.download('https://drive.google.com/uc?export=download&id=12BPy4upY6XvjPtS-gV5ZqPfo4QJVTfRC',
               'bitcoin_lstm_model.keras')

# XGBoost Model
gdown.download('https://drive.google.com/uc?export=download&id=1spdq1y3QmamtxKrJRsmHThLsCbQLCyXR',
               'bitcoin_xgb_model.pkl')

# Random Forest Model
gdown.download('https://drive.google.com/uc?export=download&id=1Cmjtw0z1XmMKkl90wLVt604ajh8mXOgH',
               'bitcoin_rf_model.pkl')

# Scalers
gdown.download('https://drive.google.com/uc?export=download&id=19MVRutA0AZ9HuCaWEAF3HtAP6T62ulOl',
               'scaler_features.pkl')
gdown.download('https://drive.google.com/uc?export=download&id=1ttQO_ULNFB0J0HoKCCtPeO4ZPiEZVAR5',
               'scaler_close.pkl')
```

## Model Performance

| Model | Accuracy | MAE | RMSE | Use Case |
|-------|----------|-----|------|----------|
| LSTM | 87.81% | $2,847 | $5,219 | Price Forecasting |
| XGBoost | 48.45% | - | - | Direction Prediction |
| Random Forest | 46.90% | - | - | Signal Confirmation |

See main [README.md](README.md) for detailed documentation.
