import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Fetch latest 1-day 1-minute RPOWER data
df = yf.download("RPOWER.NS", interval="1m", period="1d")
df.dropna(inplace=True)

# Step 2: Use only the 'Close' price
close_prices = df['Close'].values.reshape(-1, 1)

# Step 3: Normalize using MinMaxScaler
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_prices)

# Step 4: Create sequences (past 60 mins ➜ next 5-min price)
X, y = [], []
time_steps = 60
target_shift = 5  # Predict price 5 mins later

for i in range(len(scaled_close) - time_steps - target_shift):
    X.append(scaled_close[i:i + time_steps])
    y.append(scaled_close[i + time_steps + target_shift - 1])  # price after 5 mins

X, y = np.array(X), np.array(y)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 5: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Step 8: Save the model
model.save("rpower_lstm_model.h5")
print("✅ Model saved as rpower_lstm_model.h5")

# Step 9: Plot actual vs predicted
pred = model.predict(X_test)
predicted = scaler.inverse_transform(pred)
actual = scaler.inverse_transform(y_test)

plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("RPOWER Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price ₹")
plt.show()
