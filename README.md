# Bitcoin Price Prediction Models 🚀

Advanced machine learning models for Bitcoin price prediction using ensemble methods and deep learning.

## 📊 Model Performance

| Model | Accuracy | Description |
|-------|----------|-------------|
| **LSTM** | 87.81% | Deep learning model for price regression |
| **XGBoost** | 48.45% | Gradient boosting for direction prediction |
| **Random Forest** | 46.90% | Ensemble method for pattern recognition |

## 🎯 Features

### Technical Indicators (18 features)
- **Moving Averages**: MA7, MA30, MA50, EMA12, EMA26
- **Momentum**: Returns, Momentum10, Trend Strength
- **Volatility**: 7-day & 14-day volatility, Bollinger Bands
- **Oscillators**: RSI, MACD, MACD Signal, MACD Histogram
- **Volume**: Volume Change, Volume Ratio

## 📁 Project Structure

```
.
├── train_and_save_model.py      # Main training script
├── run_model.py                 # Quick evaluation script
├── bitcoin_lstm_model.keras     # Trained LSTM model
├── bitcoin_xgb_model.pkl        # Trained XGBoost model
├── bitcoin_rf_model.pkl         # Trained Random Forest model
├── scaler_features.pkl          # Feature scaler
├── scaler_close.pkl             # Price scaler
└── notebook_final.ipynb         # Research notebook
```

## 🚀 Quick Start

### Installation

```bash
pip install numpy pandas yfinance scikit-learn xgboost tensorflow joblib
```

### Training

```bash
python train_and_save_model.py
```

### Prediction

```python
import joblib
from tensorflow.keras.models import load_model

# Load models
lstm_model = load_model('bitcoin_lstm_model.keras')
xgb_model = joblib.load('bitcoin_xgb_model.pkl')
rf_model = joblib.load('bitcoin_rf_model.pkl')

# Load scalers
scaler_features = joblib.load('scaler_features.pkl')
scaler_close = joblib.load('scaler_close.pkl')
```

## 🔬 Research Highlights

- **12 years** of historical Bitcoin data (2013-2025)
- **4,200+ samples** after feature engineering
- **90-day lookback** window for LSTM sequences
- **80/20 train-test split** for validation
- **Early stopping & learning rate reduction** to prevent overfitting

## 📈 Model Details

### LSTM Architecture
- **Input**: 90-day sequences × 7 features
- **Layers**: 3 LSTM layers (128→64→32) with Dropout (0.2)
- **Output**: Price regression
- **Loss**: Huber loss (robust to outliers)

### XGBoost Configuration
- **500 trees**, max depth 5, learning rate 0.05
- **Class balancing** for imbalanced target
- **Early stopping** on validation set

### Random Forest Configuration
- **500 estimators**, max depth 8
- **Min samples split**: 20
- **Balanced class weights**

## 🎓 Use Cases

1. **Price Forecasting**: Predict future BTC prices
2. **Trading Signals**: Generate buy/sell signals
3. **Risk Management**: Volatility-based position sizing
4. **Pattern Recognition**: Identify market regimes

## ⚠️ Disclaimer

This project is for educational and research purposes only. Cryptocurrency markets are highly volatile and unpredictable. Past performance does not guarantee future results. Always do your own research and consult financial advisors before making investment decisions.

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{bitcoin_ml_2025,
  title={Bitcoin Price Prediction using Ensemble and Deep Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/bitcoin-prediction}
}
```

## 📧 Contact

- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 📄 License

MIT License - Feel free to use for research and education

---

🤖 Built with Claude Code | ⚡ Powered by TensorFlow & XGBoost
