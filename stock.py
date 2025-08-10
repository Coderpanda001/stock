import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Try importing load_model
try:
    from keras.models import load_model
except ImportError as e:
    st.error("❌ TensorFlow/Keras is not installed. Install with: pip install tensorflow-cpu==2.14.0 keras")
    st.stop()

# Title
st.title("📈 Stock Price Predictor App")

# Stock input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOG)", "GOOG")

# Load the trained model
model_path = "stock.keras"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found. Please place it in the same folder as this script.")
    st.stop()

model = load_model(model_path)

# Download stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
data = yf.download(stock, start, end)

if data.empty:
    st.error("❌ No stock data found for this symbol.")
    st.stop()

st.subheader("Stock Data")
st.write(data)

# Train/test split
split_idx = int(len(data) * 0.7)
test_data = pd.DataFrame(data.Close[split_idx:])

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=False, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving averages
for days in [250, 200, 100]:
    st.subheader(f"Close Price and {days}-Day MA")
    data[f"MA_for_{days}_days"] = data.Close.rolling(days).mean()
    st.pyplot(plot_graph((15, 6), data[f"MA_for_{days}_days"], data))

# MA comparison
st.subheader("MA for 100 days vs 250 days")
st.pyplot(plot_graph((15, 6), data["MA_for_100_days"], data, True, data["MA_for_250_days"]))

# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(test_data[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predictions
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_data = scaler.inverse_transform(y_data)

# Results
results = pd.DataFrame(
    {"Original": inv_y_data.reshape(-1), "Predicted": inv_predictions.reshape(-1)},
    index=data.index[split_idx + 100:]
)

st.subheader("Original vs Predicted Prices")
st.write(results)

# Plot Original vs Predicted
st.subheader("Close Price vs Predicted Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([data.Close[:split_idx + 100], results], axis=0))
plt.legend(["Training Data", "Original Test Data", "Predicted Data"])
st.pyplot(fig)
