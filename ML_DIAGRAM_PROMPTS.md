# ML Diagram Prompts 🎨

## For Architecture Diagrams (Draw.io, Lucidchart, Excalidraw)

### Prompt 1: Complete ML Pipeline Diagram

```
Create a comprehensive ML pipeline diagram for Bitcoin price prediction system with these components:

**Data Layer:**
- Bitcoin Data Source (Yahoo Finance API)
- 12 years historical data (2013-2025)
- 4,200+ samples

**Feature Engineering:**
- Technical Indicators (18 features)
  * Moving Averages: MA7, MA30, MA50, EMA12, EMA26
  * Momentum: Returns, Momentum10, Trend Strength
  * Volatility: 7-day, 14-day volatility
  * Oscillators: RSI, MACD, MACD Signal, MACD Histogram
  * Bollinger Bands: Width, Position %
  * Volume: Volume Change, Volume Ratio

**Data Processing:**
- Train/Test Split (80/20)
- MinMaxScaler normalization
- 90-day sequence generation for LSTM

**Model Architecture (3 models):**

1. LSTM Model (87.81% accuracy):
   - Input: 90-day × 7 features
   - Layer 1: LSTM(128) + Dropout(0.2)
   - Layer 2: LSTM(64) + Dropout(0.2)
   - Layer 3: LSTM(32) + Dropout(0.2)
   - Dense(32, ReLU)
   - Output: Dense(1) - Price Regression
   - Loss: Huber Loss
   - Optimizer: Adam

2. XGBoost Model (48.45% accuracy):
   - 500 trees, max_depth=5
   - Learning rate: 0.05
   - Subsample: 0.8
   - Output: Binary Classification (Up/Down)

3. Random Forest Model (46.90% accuracy):
   - 500 estimators, max_depth=8
   - Min samples split: 20
   - Balanced class weights
   - Output: Binary Classification (Up/Down)

**Prediction Output:**
- Price Forecast (LSTM)
- Direction Signal (XGBoost/RF)
- Confidence Score

**Evaluation Metrics:**
- LSTM: MAE, RMSE, MAPE
- Classification: Accuracy, Precision, Recall

Use modern ML colors:
- Data Layer: Blue (#2196F3)
- Feature Engineering: Green (#4CAF50)
- Processing: Orange (#FF9800)
- Models: Purple (#9C27B0)
- Output: Red (#F44336)

Include arrows showing data flow and feedback loops.
```

---

## For Flowchart Diagrams (Mermaid, PlantUML)

### Prompt 2: Mermaid Flowchart

```
Create a Mermaid flowchart for Bitcoin ML prediction pipeline:

graph TB
    A[Yahoo Finance API] --> B[Download 12Y BTC Data]
    B --> C[Feature Engineering<br/>18 Technical Indicators]
    C --> D{Data Preparation}
    D --> E[Train Split 80%]
    D --> F[Test Split 20%]

    E --> G[MinMaxScaler]
    F --> G

    G --> H[90-day Sequences]
    H --> I[LSTM Model<br/>87.81% Acc]
    H --> J[XGBoost<br/>48.45% Acc]
    H --> K[Random Forest<br/>46.90% Acc]

    I --> L[Price Prediction]
    J --> M[Direction Up/Down]
    K --> M

    L --> N[Final Forecast]
    M --> N
    N --> O[Trading Signal]

    style I fill:#9C27B0,color:#fff
    style J fill:#FF9800,color:#fff
    style K fill:#4CAF50,color:#fff
    style N fill:#F44336,color:#fff
```

---

## For Neural Network Diagrams (NN-SVG, PlotNeuralNet)

### Prompt 3: LSTM Architecture Visualization

```
Create a detailed LSTM neural network architecture diagram:

**Input Layer:**
- Shape: (90, 7) - 90 timesteps × 7 features
- Features: Close, Returns, MA7, MA30, RSI, MACD, Volatility

**Hidden Layers:**
1. LSTM Layer 1:
   - Units: 128
   - Return sequences: True
   - Activation: tanh/sigmoid gates
   - Dropout: 0.2

2. LSTM Layer 2:
   - Units: 64
   - Return sequences: True
   - Activation: tanh/sigmoid gates
   - Dropout: 0.2

3. LSTM Layer 3:
   - Units: 32
   - Return sequences: False
   - Activation: tanh/sigmoid gates
   - Dropout: 0.2

4. Dense Layer:
   - Units: 32
   - Activation: ReLU

**Output Layer:**
- Units: 1
- Activation: Linear (regression)
- Output: Predicted BTC Price

**Training Config:**
- Loss: Huber Loss
- Optimizer: Adam
- Batch Size: 32
- Epochs: 30 (with early stopping)
- Validation Split: 10%

Show cell states, hidden states, and dropout connections.
Use gradient colors showing information flow through time.
```

---

## For Comparison Diagrams (Figma, Canva)

### Prompt 4: Model Comparison Infographic

```
Create a professional infographic comparing 3 Bitcoin ML models:

**Title:** "Bitcoin Price Prediction - Model Performance"

**Model 1: LSTM Deep Learning**
- Icon: Brain/Neural Network
- Accuracy: 87.81%
- Type: Regression
- Strengths:
  * Captures temporal patterns
  * Long-term dependencies
  * Sequential data mastery
- Best for: Price forecasting
- Size: 1.6 MB

**Model 2: XGBoost**
- Icon: Decision Tree
- Accuracy: 48.45%
- Type: Classification
- Strengths:
  * Fast inference
  * Feature importance
  * Gradient boosting
- Best for: Direction prediction
- Size: 1.1 MB

**Model 3: Random Forest**
- Icon: Forest/Trees
- Accuracy: 46.90%
- Type: Classification
- Strengths:
  * Ensemble learning
  * Robust to overfitting
  * Parallel processing
- Best for: Signal confirmation
- Size: 5.4 MB

**Performance Comparison Bar Chart:**
- LSTM: 87.81% (Green, tallest)
- XGBoost: 48.45% (Orange, medium)
- Random Forest: 46.90% (Blue, medium)

**Use Case Matrix:**
- Day Trading → XGBoost ⭐⭐⭐
- Swing Trading → LSTM ⭐⭐⭐
- Long-term Investing → LSTM ⭐⭐⭐
- Risk Management → Random Forest ⭐⭐

Design style: Modern, clean, professional
Colors: Purple gradient background, white cards
Include icons and data visualization elements
```

---

## For Data Flow Diagrams (Whimsical, FigJam)

### Prompt 5: End-to-End Data Flow

```
Create an end-to-end data flow diagram:

**Phase 1: Data Collection**
[Yahoo Finance] → [API Call] → [Raw BTC Data]
- Timeframe: 2013-2025 (12 years)
- Frequency: Daily OHLCV
- Rows: 4,249

**Phase 2: Feature Engineering**
[Raw Data] → [Technical Analysis] → [Engineered Features]
- MA Calculations
- RSI, MACD computation
- Bollinger Bands
- Volume indicators
- Output: 24 columns

**Phase 3: Data Preparation**
[Features] → [Clean & Split] → [Train/Test Sets]
- Remove NaN
- 80/20 split
- Normalize (0-1)
- Sequence generation

**Phase 4: Model Training**
[Training Data] → [3 Models in Parallel]
├─ [LSTM] → [Training 30 epochs] → [Best Weights]
├─ [XGBoost] → [Training 500 trees] → [Model PKL]
└─ [Random Forest] → [Training 500 trees] → [Model PKL]

**Phase 5: Prediction**
[New Data] → [Feature Engineering] → [Normalize] → [Models]
└─ [Ensemble Output] → [Final Prediction]

**Phase 6: Deployment**
[Models] → [Google Drive] → [Web Application]
          ↓
    [REST API]
          ↓
    [User Interface]

Add icons, color coding, and time estimates for each phase.
Show parallel processing where applicable.
```

---

## For UML Diagrams (PlantUML)

### Prompt 6: Class Diagram

```
@startuml
class BitcoinDataLoader {
  - symbol: str
  - start_date: datetime
  - end_date: datetime
  + download_data(): DataFrame
  + validate_data(): bool
}

class FeatureEngineer {
  - df: DataFrame
  + calc_rsi(window: int): Series
  + calc_ma(window: int): Series
  + calc_macd(): tuple
  + calc_bollinger_bands(): tuple
  + generate_features(): DataFrame
}

class DataPreprocessor {
  - train_size: float
  + split_data(): tuple
  + normalize(scaler: MinMaxScaler): array
  + create_sequences(seq_len: int): tuple
}

class LSTMModel {
  - input_shape: tuple
  - layers: list
  + build_model(): Sequential
  + train(X, y, epochs: int): History
  + predict(X): array
  + save_model(path: str): void
}

class XGBoostModel {
  - n_estimators: int
  - max_depth: int
  + train(X, y): void
  + predict(X): array
  + get_feature_importance(): dict
}

class RandomForestModel {
  - n_estimators: int
  - max_depth: int
  + train(X, y): void
  + predict(X): array
  + get_feature_importance(): dict
}

class ModelEvaluator {
  + calculate_accuracy(y_true, y_pred): float
  + calculate_mae(y_true, y_pred): float
  + calculate_rmse(y_true, y_pred): float
  + plot_results(): void
}

class PredictionEngine {
  - lstm_model: LSTMModel
  - xgb_model: XGBoostModel
  - rf_model: RandomForestModel
  + ensemble_predict(X): dict
  + get_confidence_score(): float
}

BitcoinDataLoader --> FeatureEngineer
FeatureEngineer --> DataPreprocessor
DataPreprocessor --> LSTMModel
DataPreprocessor --> XGBoostModel
DataPreprocessor --> RandomForestModel
LSTMModel --> ModelEvaluator
XGBoostModel --> ModelEvaluator
RandomForestModel --> ModelEvaluator
LSTMModel --> PredictionEngine
XGBoostModel --> PredictionEngine
RandomForestModel --> PredictionEngine
@enduml
```

---

## For Timeline Diagrams (Timeline.js, Visme)

### Prompt 7: Research & Development Timeline

```
Create a project timeline showing ML model development:

**Week 1: Research & Data Collection**
- Literature review on Bitcoin prediction
- Yahoo Finance API integration
- Downloaded 12 years of data
- Exploratory data analysis

**Week 2: Feature Engineering**
- Implemented 18 technical indicators
- Tested different feature combinations
- Feature importance analysis
- Correlation matrix

**Week 3: Model Development - Classification**
- Random Forest baseline (46.90%)
- XGBoost optimization (48.45%)
- Hyperparameter tuning
- Cross-validation

**Week 4: Model Development - Regression**
- LSTM architecture design
- Sequence generation (90-day lookback)
- Training with early stopping
- Achieved 87.81% accuracy

**Week 5: Model Evaluation**
- Performance metrics calculation
- Model comparison analysis
- Threshold optimization
- Confidence scoring

**Week 6: Deployment**
- Model serialization
- Google Drive upload
- GitHub repository setup
- Documentation

Mark milestones with accuracy improvements and breakthroughs.
Use color gradient showing progress from red (start) to green (completion).
```

---

## Quick Tips for Diagram Creation

### For Claude/ChatGPT:
1. Copy any prompt above
2. Add: "Generate this as [Mermaid/PlantUML/SVG code]"
3. Paste output into diagram tool

### For Diagram Tools:
- **Mermaid Live**: mermaid.live
- **PlantUML Online**: plantuml.com
- **Draw.io**: app.diagrams.net
- **Excalidraw**: excalidraw.com
- **Whimsical**: whimsical.com

### Color Palette:
```
Primary: #9C27B0 (Purple) - LSTM/Deep Learning
Secondary: #FF9800 (Orange) - XGBoost
Accent: #4CAF50 (Green) - Random Forest
Background: #F5F5F5 (Light Gray)
Text: #212121 (Dark Gray)
Success: #4CAF50 (Green)
Warning: #FFC107 (Amber)
Error: #F44336 (Red)
```

---

🎨 **Pro Tip:** For LinkedIn posts, use infographic-style diagrams (Prompt 4) as they get the most engagement. For technical documentation, use architecture diagrams (Prompt 1) or flowcharts (Prompt 2).
