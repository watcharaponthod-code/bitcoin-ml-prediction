import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

print("="*60)
print("Bitcoin Price Prediction - Model Training")
print("="*60)

# Download Data
symbol = 'BTC-USD'
end_date = datetime.now()
start_date = end_date - timedelta(days=12*365)
btc_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
print(f"✓ Downloaded {len(btc_data)} rows of data")

# Feature Engineering
def calc_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if isinstance(btc_data.columns, pd.MultiIndex):
    btc_data.columns = btc_data.columns.get_level_values(0)

close = btc_data['Close']
volume = btc_data['Volume']

btc_data['Returns'] = close.pct_change()
btc_data['MA7'] = close.rolling(7).mean()
btc_data['MA30'] = close.rolling(30).mean()
btc_data['MA50'] = close.rolling(50).mean()
btc_data['EMA12'] = close.ewm(span=12, adjust=False).mean()
btc_data['EMA26'] = close.ewm(span=26, adjust=False).mean()
btc_data['Momentum10'] = close.pct_change(10)
btc_data['Trend_Strength'] = (btc_data['MA7'] - btc_data['MA30']) / btc_data['MA30']
btc_data['Volatility'] = btc_data['Returns'].rolling(7).std()
btc_data['Volatility14'] = btc_data['Returns'].rolling(14).std()
btc_data['RSI'] = calc_rsi(close)
btc_data['MACD'] = btc_data['EMA12'] - btc_data['EMA26']
btc_data['MACD_Signal'] = btc_data['MACD'].ewm(span=9, adjust=False).mean()
btc_data['MACD_Hist'] = btc_data['MACD'] - btc_data['MACD_Signal']
bb_ma20 = close.rolling(20).mean()
bb_std20 = close.rolling(20).std()
btc_data['BB_Width'] = (4 * bb_std20) / bb_ma20
btc_data['BB_Pct'] = (close - (bb_ma20 - 2*bb_std20)) / (4 * bb_std20)
btc_data['Volume_Change'] = volume.pct_change()
vol_ma20 = volume.rolling(20).mean()
btc_data['Volume_Ratio'] = volume / vol_ma20
btc_data['Target'] = (close.shift(-7) > close).astype(int)

btc_prepared = btc_data.dropna().copy()
print(f"✓ Prepared {len(btc_prepared)} rows with {btc_prepared.shape[1]} features")

# Train Models
features = [
    'Returns', 'MA7', 'MA30', 'MA50',
    'EMA12', 'EMA26', 'Momentum10', 'Trend_Strength',
    'Volatility', 'Volatility14',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Pct',
    'Volume_Change', 'Volume_Ratio'
]
X = btc_prepared[features]
y = btc_prepared['Target']

split_index = int(len(btc_prepared) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Random Forest
print("\n" + "="*60)
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_split=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
print(f"✓ Random Forest Accuracy: {rf_acc:.2f}%")
joblib.dump(rf, 'bitcoin_rf_model.pkl')
print("✓ Saved: bitcoin_rf_model.pkl")

# XGBoost
print("\n" + "="*60)
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
    eval_metric='logloss', random_state=42
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test)) * 100
print(f"✓ XGBoost Accuracy: {xgb_acc:.2f}%")
joblib.dump(xgb, 'bitcoin_xgb_model.pkl')
print("✓ Saved: bitcoin_xgb_model.pkl")

# LSTM
print("\n" + "="*60)
print("Training LSTM...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

lstm_features = ['Close', 'Returns', 'MA7', 'MA30', 'RSI', 'MACD', 'Volatility']
scaler_reg = MinMaxScaler(feature_range=(0, 1))
scaled_reg = scaler_reg.fit_transform(btc_prepared[lstm_features].values)
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler_close.fit_transform(btc_prepared[['Close']].values)

SEQ_LEN = 90
X_lstm, y_lstm = [], []
for i in range(SEQ_LEN, len(scaled_reg)):
    X_lstm.append(scaled_reg[i-SEQ_LEN:i])
    y_lstm.append(scaled_close[i, 0])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
_split = int(len(X_lstm) * 0.8)
X_tr_l, X_te_l = X_lstm[:_split], X_lstm[_split:]
y_tr_l, y_te_l = y_lstm[:_split], y_lstm[_split:]

model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(lstm_features))),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='huber')

cb_reg = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
]
history = model_lstm.fit(X_tr_l, y_tr_l, batch_size=32, epochs=30,
                         validation_split=0.1, callbacks=cb_reg, verbose=0)

preds_reg = scaler_close.inverse_transform(model_lstm.predict(X_te_l, verbose=0))
actual_reg = scaler_close.inverse_transform(y_te_l.reshape(-1,1))
mae = mean_absolute_error(actual_reg, preds_reg)
rmse = np.sqrt(mean_squared_error(actual_reg, preds_reg))
mape = np.mean(np.abs((actual_reg - preds_reg) / actual_reg)) * 100
acc_reg = 100 - mape
print(f"✓ LSTM Accuracy (100-MAPE): {acc_reg:.2f}%")

model_lstm.save('bitcoin_lstm_model.keras')
print("✓ Saved: bitcoin_lstm_model.keras")
joblib.dump(scaler_reg, 'scaler_features.pkl')
joblib.dump(scaler_close, 'scaler_close.pkl')
print("✓ Saved scalers: scaler_features.pkl, scaler_close.pkl")

print("\n" + "="*60)
print("All Models Saved Successfully!")
print("="*60)
print(f"Random Forest: {rf_acc:.2f}%")
print(f"XGBoost:       {xgb_acc:.2f}%")
print(f"LSTM:          {acc_reg:.2f}%")
