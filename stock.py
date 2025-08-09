import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

@st.cache_resource
def load_trained_model(path="stockpredict.keras"):
    # Wrap in try/except to surface friendly error if the file is missing/corrupt
    try:
        m = load_model(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_trained_model("stockpredict.keras")

def predict_and_suggest_action(data_test_scale, scaler, window_size, model):
    x = []
    y = []
    n = data_test_scale.shape[0]
    if n <= window_size:
        return np.array([]), np.array([])

    for i in range(window_size, n):
        x.append(data_test_scale[i - window_size:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    if x.shape[0] == 0:
        return np.array([]), np.array([])

    predict = model.predict(x, verbose=0)

    # Inverse transform
    predict = scaler.inverse_transform(predict)
    y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    return predict.flatten(), y

def suggest_action(predicted_price, current_price):
    return "Buy" if predicted_price > current_price else "Sell"

def main():
    st.title('📈 Stock Market Dashboard')
    st.markdown("---")

    if model is None:
        st.stop()

    # Validate model input shape for a single 3D sequence input (batch, timesteps, features)
    try:
        inp_shape = model.input_shape
        # handle models with multiple inputs by picking the first
        if isinstance(inp_shape, list):
            inp_shape = inp_shape[0]
        # Expect (None, window_size, features)
        window_size = inp_shape[1]
        n_features = inp_shape[2] if len(inp_shape) > 2 else 1
        if window_size is None or n_features is None or n_features < 1:
            st.error("Model input shape is invalid or undefined. Expected 3D input.")
            st.stop()
    except Exception as e:
        st.error(f"Could not determine model input shape: {e}")
        st.stop()

    # Date range
    st.subheader("Select Data Range to Predict")
    default_start = pd.to_datetime('2012-01-01')
    default_end = pd.to_datetime(datetime.now().date())
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

    # Convert to pandas Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date >= end_date:
        st.warning("Start Date must be before End Date.")
        st.stop()

    # Because yfinance end is exclusive, add 1 day to include the chosen end_date
    yf_end = end_date + pd.Timedelta(days=1)

    # Select stock
    selected_stock = st.selectbox('Select Stock Symbol', ['AAPL', 'GOOG', 'MSFT', 'AMZN'])

    # Download data
    try:
        data = yf.download(selected_stock, start=start_date, end=yf_end, auto_adjust=False, progress=False)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if data is None or data.empty:
        st.error("No data found for the selected range/symbol. Try a different date range or stock.")
        st.stop()

    # Ensure Close exists
    if 'Close' not in data.columns:
        st.error("Downloaded data does not contain 'Close' prices.")
        st.stop()

    st.write(data)

    # Train/test split
    split_idx = int(len(data) * 0.80)
    data_train = pd.DataFrame(data['Close'].iloc[:split_idx]).reset_index(drop=True)
    data_test = pd.DataFrame(data['Close'].iloc[split_idx:]).reset_index(drop=True)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    try:
        scaler.fit(data_train)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        st.stop()

    # Prepare test with past window
    if len(data_train) < window_size:
        st.error(f"Not enough training data for the model window size ({window_size}). Increase date range.")
        st.stop()

    pas_days = data_train.tail(window_size)
    data_test_full = pd.concat([pas_days, data_test], ignore_index=True)

    # Model expects features dimension; our series has shape (n, 1)
    try:
        data_test_scale = scaler.transform(data_test_full)
    except Exception as e:
        st.error(f"Scaling transform failed: {e}")
        st.stop()

    # Prediction
    predict, y = predict_and_suggest_action(data_test_scale, scaler, window_size, model)

    # Suggested action based on last predicted vs last actual close
    last_close = float(data['Close'].iloc[-1])
    if predict.size > 0:
        suggested_action = suggest_action(float(predict[-1]), last_close)
    else:
        suggested_action = None

    # ANALYSIS
    st.subheader('🔍 Analysis')

    st.subheader('📉 Original Price vs Predicted Price')
    if predict.size == 0 or y.size == 0:
        st.info("Not enough data to generate predictions with the current window size and date range.")
    else:
        fig, ax = plt.subplots()
        ax.plot(y, 'r', label='Original Price')
        ax.plot(predict, 'g', label='Predicted Price')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader('🧮 Predicted vs Actual Values')
        results = pd.DataFrame({'Predicted': predict, 'Actual': y})
        st.write(results)

    st.subheader('📜 Historical Closing Prices')
    fig_close, ax_close = plt.subplots()
    ax_close.plot(data['Close'], label='Closing Price')
    ax_close.set_xlabel('Date')
    ax_close.set_ylabel('Price')
    ax_close.legend()
    st.pyplot(fig_close)
    plt.close(fig_close)

    st.subheader('📊 Moving Averages')
    data_ma = data.copy()
    data_ma['MA50'] = data_ma['Close'].rolling(window=50).mean()
    data_ma['MA200'] = data_ma['Close'].rolling(window=200).mean()
    fig_ma, ax_ma = plt.subplots()
    ax_ma.plot(data_ma['Close'], label='Closing Price')
    ax_ma.plot(data_ma['MA50'], label='50-Day MA')
    ax_ma.plot(data_ma['MA200'], label='200-Day MA')
    ax_ma.set_xlabel('Date')
    ax_ma.set_ylabel('Price')
    ax_ma.legend()
    st.pyplot(fig_ma)
    plt.close(fig_ma)

    st.subheader('⚠️ Volatility')
    data_vol = data.copy()
    data_vol['Returns'] = data_vol['Close'].pct_change()
    data_vol['Volatility'] = data_vol['Returns'].rolling(window=50).std() * np.sqrt(50)
    fig_vol, ax_vol = plt.subplots()
    ax_vol.plot(data_vol['Volatility'], label='Volatility')
    ax_vol.set_xlabel('Date')
    ax_vol.set_ylabel('Volatility')
    ax_vol.legend()
    st.pyplot(fig_vol)
    plt.close(fig_vol)

    st.subheader('✅ Suggested Action')
    if suggested_action is None:
        st.info("No suggestion available because predictions were not generated.")
    else:
        if suggested_action == "Buy":
            st.success(f"Suggested Action: {suggested_action}")
        else:
            st.error(f"Suggested Action: {suggested_action}")

    st.markdown("---")
    st.markdown("Contact Us / support:")
    st.markdown("- click here : https://tradelitcare.streamlit.app ")

    st.markdown("---")
    st.write(
        """
        <div style="overflow-x: auto; white-space: nowrap;">
            <marquee behavior="scroll" direction="left" scrollamount="5">
                Intraday Data provided by FACTSET and subject to terms of use. 
                Historical and current end-of-day data provided by FACTSET. 
                All quotes are in local exchange time. Real-time last sale data for U.S. 
                stock quotes reflect trades reported through Nasdaq only. 
                Intraday. data delayed at least 15 minutes or per exchange requirements.
            </marquee>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
