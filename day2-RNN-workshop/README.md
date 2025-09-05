# Day 2: RNN Stock Price Prediction

## Objective
Build and train a Recurrent Neural Network (LSTM) to predict Tesla stock prices.

## Steps
1. Load the Tesla stock price dataset (from Kaggle).
2. Preprocess the dataset (scaling, sequence framing, train-test split).
3. Build and train an RNN model (LSTM).
4. Evaluate using regression metrics (MSE, MAE, R2 score).
5. Plot real vs predicted stock prices.

---
**Ajit Kumar**

# Core Concept Questions & Answers

### 1. What is the benefit of using RNNs (or LSTMs) over traditional feedforward networks for time-series data?

Traditional feedforward neural networks process inputs independently and have no inherent memory of past values, which makes them unsuitable for sequential or temporal data such as stock prices. Recurrent Neural Networks (RNNs) are specifically designed to process sequences, as they have internal loops that allow them to "remember" information from previous time steps. This enables them to capture dependencies across time. LSTMs (Long Short-Term Memory networks), in particular, extend RNNs by using gating mechanisms (input, forget, and output gates) to better handle long-term dependencies and mitigate the vanishing gradient problem. This makes LSTMs highly effective for stock price prediction, where past price movements strongly influence future values.

### 2. Why is sequence framing (input windows) important in time series forecasting?

Sequence framing transforms time-series data into a supervised learning problem. For example, instead of predicting tomorrow’s stock price directly from a single day’s price, we use the last n days (like a sliding window of 60 days) as input to predict the next day. This is essential because stock prices depend heavily on trends, momentum, and seasonality in past data. Without framing, the model would lose temporal context. Sequence framing essentially teaches the model to “learn patterns over time” and makes predictions more realistic for real-world forecasting problems.

### 3. How does feature scaling impact training of RNN/LSTM models?

Feature scaling ensures that input data values are within a similar numerical range, usually between 0 and 1. Since stock prices can range from tens to thousands of dollars, unscaled data could bias the training process, making optimization unstable and slower. Scaling also prevents one feature from dominating others due to its magnitude. More importantly, for RNNs and LSTMs, feature scaling reduces the risk of exploding or vanishing gradients during backpropagation through time. As a result, training becomes more efficient, convergence is faster, and predictions are more accurate.

### 4. Compare SimpleRNN and LSTM in terms of handling long-term dependencies.

SimpleRNN: It is the basic form of RNN and works well for short sequences, but struggles with long-term dependencies because gradients diminish as sequences get longer (the vanishing gradient problem). This limits its usefulness for tasks like stock price prediction, where long-term patterns matter.

LSTM: It improves upon RNN by using specialized gates that decide which information to keep, forget, or output. This allows it to retain information over long sequences and effectively learn long-term relationships. For example, LSTMs can detect trends in stock prices spanning several weeks or months, something SimpleRNN cannot handle well. Thus, LSTMs are more robust for real-world time-series forecasting.

### 5. What regression metrics (e.g., MAE, RMSE) are appropriate for stock price prediction, and why?

For stock price prediction, regression metrics are essential because the task involves predicting continuous values.

MAE (Mean Absolute Error): Provides the average absolute difference between predicted and actual prices, which is easy to interpret in financial terms (e.g., “on average, predictions are off by $5”).

RMSE (Root Mean Squared Error): Punishes larger errors more severely, which is useful in finance where large mistakes can have big consequences.

R² (Coefficient of Determination): Explains how much variance in stock prices is captured by the model. Higher values mean the model is capturing more useful trends.
Using multiple metrics provides a balanced view: MAE for interpretability, RMSE for error severity, and R² for explanatory power.

### 6. How can you assess if your model is overfitting?

Overfitting occurs when the model performs very well on the training set but fails to generalize on unseen test data. Some ways to detect overfitting include:

Training loss decreases steadily, but test loss starts increasing.

Predictions on training data are highly accurate, but predictions on new data are poor.

Visualizing predictions vs. actual values shows that the model fits training patterns too closely but fails to follow test trends.

Performance metrics like MAE and RMSE are much better on training data compared to test data.
This means the model is memorizing noise rather than learning general patterns.

### 7. How might you extend the model to improve performance (e.g., more features, deeper network)?

Several strategies can improve performance:

Add more features: Instead of only using the "Open" price, include "High", "Low", "Close", "Volume", or even technical indicators like Moving Averages (MA), RSI, MACD.

Deeper network: Stack multiple LSTM layers to increase the capacity of the network for learning complex patterns.

Regularization: Apply dropout layers, early stopping, or L2 regularization to prevent overfitting.

Hyperparameter tuning: Adjust learning rate, batch size, sequence length, and number of units.

Data augmentation: Train on longer historical data or incorporate external features like news sentiment.
These enhancements make the model more robust and better aligned with real-world stock prediction challenges.

### 8. Why is it important to shuffle (or not shuffle) sequential data during training?

In standard machine learning tasks, shuffling improves generalization by breaking order bias. However, for time-series forecasting, shuffling must not be done because order matters. Stock prices depend on sequential history, and breaking the order would destroy temporal relationships. Instead, we train the model on sequential batches (in order), but we can still shuffle within training samples of the same sequence length. This ensures the model respects temporal order while still improving training efficiency.

### 9. How can you visualize model prediction vs actual values to interpret performance?

The most effective way is to plot the predicted stock prices against the actual stock prices over time. By overlaying the two curves, you can visually inspect how closely the model follows the real trends. If the predicted line tracks the real prices well, the model is performing effectively. Additionally, scatter plots of predicted vs actual values or error distribution plots can reveal bias, variance, and systematic prediction issues.

### 10. What real-world challenges arise when using RNNs for stock price prediction?

Noise and randomness: Stock prices are influenced by many unpredictable factors like news, political events, or investor sentiment, which no model can perfectly capture.

Overfitting: The model might fit historical data patterns that don’t generalize to future markets.

Data limitations: Historical prices may not provide enough context; external features (macroeconomic indicators, news, global events) may be required.

Changing patterns: Market conditions evolve (COVID-19 crash, policy changes), making old patterns less useful.

Computational cost: Training deep RNNs/LSTMs can be expensive and time-consuming.
Despite these challenges, careful preprocessing, feature engineering, and robust evaluation can make RNNs valuable tools for forecasting.
