# Bitcoin ML Models - Download Links 🚀

## 📁 Google Drive Folder
**All Models:** https://drive.google.com/drive/folders/1qpKcXxEf4aPNdUxSPRWnJ_0kVbXAlBt1

## 📦 Individual Model Files

### 1. LSTM Model (87.81% accuracy)
- **File:** bitcoin_lstm_model.keras (1.6 MB)
- **View:** https://drive.google.com/file/d/12BPy4upY6XvjPtS-gV5ZqPfo4QJVTfRC/view
- **Download:** https://drive.google.com/uc?export=download&id=12BPy4upY6XvjPtS-gV5ZqPfo4QJVTfRC

### 2. XGBoost Model (48.45% accuracy)
- **File:** bitcoin_xgb_model.pkl (1.1 MB)
- **View:** https://drive.google.com/file/d/1spdq1y3QmamtxKrJRsmHThLsCbQLCyXR/view
- **Download:** https://drive.google.com/uc?export=download&id=1spdq1y3QmamtxKrJRsmHThLsCbQLCyXR

### 3. Random Forest Model (46.90% accuracy)
- **File:** bitcoin_rf_model.pkl (5.4 MB)
- **View:** https://drive.google.com/file/d/1Cmjtw0z1XmMKkl90wLVt604ajh8mXOgH/view
- **Download:** https://drive.google.com/uc?export=download&id=1Cmjtw0z1XmMKkl90wLVt604ajh8mXOgH

### 4. Feature Scaler
- **File:** scaler_features.pkl (959 bytes)
- **View:** https://drive.google.com/file/d/19MVRutA0AZ9HuCaWEAF3HtAP6T62ulOl/view
- **Download:** https://drive.google.com/uc?export=download&id=19MVRutA0AZ9HuCaWEAF3HtAP6T62ulOl

### 5. Price Scaler
- **File:** scaler_close.pkl (719 bytes)
- **View:** https://drive.google.com/file/d/1ttQO_ULNFB0J0HoKCCtPeO4ZPiEZVAR5/view
- **Download:** https://drive.google.com/uc?export=download&id=1ttQO_ULNFB0J0HoKCCtPeO4ZPiEZVAR5

## 🔧 How to Use

### Download All Models
```python
import gdown

# LSTM Model
gdown.download('https://drive.google.com/uc?export=download&id=12BPy4upY6XvjPtS-gV5ZqPfo4QJVTfRC', 'bitcoin_lstm_model.keras')

# XGBoost Model
gdown.download('https://drive.google.com/uc?export=download&id=1spdq1y3QmamtxKrJRsmHThLsCbQLCyXR', 'bitcoin_xgb_model.pkl')

# Random Forest Model
gdown.download('https://drive.google.com/uc?export=download&id=1Cmjtw0z1XmMKkl90wLVt604ajh8mXOgH', 'bitcoin_rf_model.pkl')

# Scalers
gdown.download('https://drive.google.com/uc?export=download&id=19MVRutA0AZ9HuCaWEAF3HtAP6T62ulOl', 'scaler_features.pkl')
gdown.download('https://drive.google.com/uc?export=download&id=1ttQO_ULNFB0J0HoKCCtPeO4ZPiEZVAR5', 'scaler_close.pkl')
```

### Load Models
```python
import joblib
from tensorflow.keras.models import load_model

# Load LSTM
lstm_model = load_model('bitcoin_lstm_model.keras')

# Load XGBoost & Random Forest
xgb_model = joblib.load('bitcoin_xgb_model.pkl')
rf_model = joblib.load('bitcoin_rf_model.pkl')

# Load Scalers
scaler_features = joblib.load('scaler_features.pkl')
scaler_close = joblib.load('scaler_close.pkl')
```

## 📊 Model Statistics

| Model | Type | Accuracy | Size | Use Case |
|-------|------|----------|------|----------|
| LSTM | Regression | 87.81% | 1.6 MB | Price prediction |
| XGBoost | Classification | 48.45% | 1.1 MB | Direction (up/down) |
| Random Forest | Classification | 46.90% | 5.4 MB | Direction (up/down) |

## 🎯 Features Used

18 technical indicators including:
- Moving Averages (MA7, MA30, MA50, EMA12, EMA26)
- Momentum Indicators (Returns, Momentum10, Trend Strength)
- Volatility Measures (7-day, 14-day)
- Oscillators (RSI, MACD)
- Bollinger Bands (Width, Position)
- Volume Analysis (Volume Change, Volume Ratio)

## 📝 Training Details

- **Dataset:** 12 years of BTC-USD data (2013-2025)
- **Samples:** 4,200+ after feature engineering
- **Train/Test Split:** 80/20
- **LSTM Lookback:** 90 days
- **Validation:** Early stopping with patience

---

🤖 Trained with Claude Code | ⚡ Powered by TensorFlow & XGBoost
