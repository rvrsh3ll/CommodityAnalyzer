```
=== Commodity Analysis Tool ===
1. Analyze a specific symbol
2. Analyze all predefined commodities
3. Exit
Enter your choice (1-3): 1
Enter the ticker symbol to analyze (e.g., AAPL, GOOGL): f
[*********************100%***********************]  1 of 1 completed

Analyzing F
Initial data shape: (686, 6)
Feature engineering: Initial shape: (686, 6)
Before dropna: (686, 34)
After dropna: (653, 34)
Feature engineering: Final shape: (653, 34)
After feature engineering shape: (653, 34)
Columns in df: Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Returns',
       'Log_Returns', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ADX',
       'CCI', 'ROC', 'MOM', 'OBV', 'STOCH_K', 'STOCH_D', 'WILLR', 'SMA_10',
       'SMA_30', 'EMA_10', 'EMA_30', 'Upper_BB', 'Middle_BB', 'Lower_BB',
       'BB_Width', 'Price_Direction_1d', 'Price_Direction_5d',
       'Price_Direction_20d', 'Day_of_Week', 'Month'],
      dtype='object')
Shape of df: (653, 34)
Shape of X: (653, 25)
Shape of y_1d: (653,)
Shape of y_5d: (653,)
Shape of y_20d: (653,)
X shape: (653, 25)
y_1d shape: (653,)
y_5d shape: (653,)
y_20d shape: (653,)

Training model for 1d horizon
Training RandomForest...
RandomForest best F1 score: 0.5478
Training GradientBoosting...
GradientBoosting best F1 score: 0.5456
Training XGBoost...

XGBoost best F1 score: 0.5207
Training LightGBM...
LightGBM best F1 score: 0.5711
Training LogisticRegression...
LogisticRegression best F1 score: 0.5902
Training model for 5d horizon
Training RandomForest...
RandomForest best F1 score: 0.5053
Training GradientBoosting...
GradientBoosting best F1 score: 0.5011
Training XGBoost...

XGBoost best F1 score: 0.5062
Training LightGBM...
LightGBM best F1 score: 0.4963
Training LogisticRegression...
LogisticRegression best F1 score: 0.5390

Training model for 20d horizon
Training RandomForest...
RandomForest best F1 score: 0.6044
Training GradientBoosting...
GradientBoosting best F1 score: 0.6472
Training XGBoost...

XGBoost best F1 score: 0.4969
Training LightGBM...
LightGBM best F1 score: 0.6140
Training LogisticRegression...
LogisticRegression best F1 score: 0.6735

Analysis for F (F):
Current Price: $10.80

1d Prediction:
Signal: BUY
Confidence: 0.03
Model Performance:
RandomForest F1 Score: 0.5478
GradientBoosting F1 Score: 0.5456
XGBoost F1 Score: 0.5207
LightGBM F1 Score: 0.5711
LogisticRegression F1 Score: 0.5902

5d Prediction:
Signal: SELL
Confidence: 0.14
Model Performance:
RandomForest F1 Score: 0.5053
GradientBoosting F1 Score: 0.5011
XGBoost F1 Score: 0.5062
LightGBM F1 Score: 0.4963
LogisticRegression F1 Score: 0.5390

20d Prediction:
Signal: SELL
Confidence: 0.13
Model Performance:
RandomForest F1 Score: 0.6044
GradientBoosting F1 Score: 0.6472
XGBoost F1 Score: 0.4969
LightGBM F1 Score: 0.6140
LogisticRegression F1 Score: 0.6735

=== Beginner-Friendly Summary ===
Analysis for F:
Current Price: $10.80

1d Prediction:
The model suggests to BUY
with 0.03 confidence.
Average Model F1 Score: 0.56

5d Prediction:
The model suggests to SELL
with 0.14 confidence.
Average Model F1 Score: 0.51

20d Prediction:
The model suggests to SELL
with 0.13 confidence.
Average Model F1 Score: 0.61

What does this mean?
- A 'BUY' suggestion means the model thinks the price might go up.
- A 'SELL' suggestion means the model thinks the price might go down.
- Confidence ranges from 0 to 1. Higher numbers mean the model is more sure.
- F1 Score is a measure of the model's accuracy. It ranges from 0 to 1, where 1 is perfect.

Risk Management Suggestions:
1. Don't make decisions based solely on this model. It's just one tool.
2. Always use stop-loss orders to limit potential losses.
3. Never invest more than you can afford to lose.
4. Consider the overall market conditions and news that might affect the asset.
5. Diversify your investments to spread risk.
6. Be aware that short-term predictions (1d) are generally less reliable than longer-term ones.
7. Monitor your positions regularly and be prepared to exit if conditions change.

Further Steps:
1. Research the fundamentals of the asset you're interested in.
2. Look at longer-term trends and overall market conditions.
3. Consider consulting with a financial advisor for personalized advice.
4. Keep learning about different investment strategies and risk management techniques.
```
