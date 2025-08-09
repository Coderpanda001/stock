import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained model
@st.cache_resource
def load_stock_model():
    try:
        model = load_model("stockpredict.keras")
        return model
    except:
        st.error("Model file 'stockpredict.keras' not found. Please ensure the model file is in the same directory.")
        return None

def download_stock_data(symbol, start_date, end_date):
    """Download stock data with proper error handling"""
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

        if data.empty:
            return None

        # Handle multi-level columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns
            data.columns = [col[0] if col[1] == symbol else col[0] + '_' + col[1] for col in data.columns]

        # Ensure we have the basic OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            st.warning(f"Missing columns: {missing_columns}")

        return data

    except Exception as e:
        st.error(f"Error downloading data for {symbol}: {str(e)}")
        return None

def predict_and_suggest_action(data_test_scale, scaler, window_size, model):
    """Make predictions with error handling"""
    if model is None:
        return None, None

    x = []
    y = []

    for i in range(window_size, data_test_scale.shape[0]):
        x.append(data_test_scale[i - window_size:i])
        y.append(data_test_scale[i, 0])

    if len(x) == 0:
        return None, None

    x, y = np.array(x), np.array(y)

    try:
        predict = model.predict(x, verbose=0)

        # Inverse transform
        predict = scaler.inverse_transform(predict)
        y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

        return predict.flatten(), y
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def suggest_action(predicted_price, current_price, threshold=0.02):
    """Suggest action based on predicted vs current price"""
    if predicted_price is None or current_price is None:
        return "Hold"

    price_change = (predicted_price - current_price) / current_price

    if price_change > threshold:
        return "Buy"
    elif price_change < -threshold:
        return "Sell"
    else:
        return "Hold"

def calculate_metrics(predicted, actual):
    """Calculate prediction metrics"""
    if predicted is None or actual is None:
        return {}

    mse = np.mean((predicted - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted - actual))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def main():
    st.set_page_config(page_title="Stock Market Dashboard", page_icon="ðŸ“ˆ", layout="wide")

    st.title('ðŸ“ˆ Stock Market Prediction Dashboard')
    st.markdown("---")

    # Load model
    model = load_stock_model()

    if model is None:
        st.stop()

    # Sidebar for inputs
    st.sidebar.header("ðŸ“Š Configuration")

    # Date range with better defaults
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime('2024-12-31'))

    # Convert to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Validate date range
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        st.stop()

    # Select stock with more options
    stock_options = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corporation',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corporation',
        'META': 'Meta Platforms Inc.',
        'NFLX': 'Netflix Inc.'
    }

    selected_symbol = st.sidebar.selectbox(
        'Select Stock Symbol', 
        list(stock_options.keys()),
        format_func=lambda x: f"{x} - {stock_options[x]}"
    )

    # Train-test split ratio
    train_ratio = st.sidebar.slider("Training Data Ratio", 0.7, 0.9, 0.8, 0.05)

    # Download data button
    if st.sidebar.button("ðŸ“¥ Download & Analyze Data"):

        with st.spinner(f"Downloading data for {selected_symbol}..."):
            data = download_stock_data(selected_symbol, start_date, end_date)

        if data is None or data.empty:
            st.error("No data found. Please try a different date range or stock symbol.")
            st.stop()

        st.success(f"Successfully downloaded {len(data)} days of data for {selected_symbol}")

        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        with col3:
            st.metric("52W High", f"${data['Close'].max():.2f}")
        with col4:
            st.metric("52W Low", f"${data['Close'].min():.2f}")

        # Prepare data for model
        try:
            split_index = int(len(data) * train_ratio)
            data_train = pd.DataFrame(data['Close'][:split_index])
            data_test = pd.DataFrame(data['Close'][split_index:])

            if len(data_train) < 100 or len(data_test) < 50:
                st.warning("Insufficient data for reliable predictions. Try a longer date range.")

            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data_train)

            # Get model window size (default to 60 if not available)
            try:
                window_size = model.input_shape[1]
            except:
                window_size = 60
                st.info(f"Using default window size of {window_size} days")

            # Prepare test data
            past_days = data_train.tail(window_size)
            data_test_full = pd.concat([past_days, data_test], ignore_index=True)
            data_test_scale = scaler.transform(data_test_full)

            # Make predictions
            with st.spinner("Making predictions..."):
                predict, y_actual = predict_and_suggest_action(data_test_scale, scaler, window_size, model)

            if predict is not None and y_actual is not None:
                # Calculate suggested action
                current_price = float(data['Close'].iloc[-1])
                predicted_price = float(predict[-1])
                suggested_action = suggest_action(predicted_price, current_price)

                # Display prediction results
                st.subheader('ðŸŽ¯ Prediction Results')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Predicted Price", f"${predicted_price:.2f}")
                with col3:
                    color = "normal"
                    if suggested_action == "Buy":
                        color = "inverse"
                    elif suggested_action == "Sell":
                        color = "off"
                    st.metric("Suggested Action", suggested_action)

                # Calculate and display metrics
                metrics = calculate_metrics(predict, y_actual)
                if metrics:
                    st.subheader('ðŸ“Š Model Performance Metrics')
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    with col2:
                        st.metric("MAE", f"{metrics['MAE']:.4f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    with col4:
                        accuracy = max(0, 100 - metrics['MAPE'])
                        st.metric("Accuracy", f"{accuracy:.2f}%")

                # Visualization
                st.subheader('ðŸ“ˆ Price Prediction Analysis')

                # Create prediction plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(range(len(y_actual)), y_actual, 'b-', label='Actual Price', linewidth=2)
                ax.plot(range(len(predict)), predict, 'r--', label='Predicted Price', linewidth=2)
                ax.set_xlabel('Trading Days')
                ax.set_ylabel('Stock Price ($)')
                ax.set_title(f'{selected_symbol} - Actual vs Predicted Prices')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

                # Historical price chart
                st.subheader('ðŸ“Š Historical Analysis')

                # Price and moving averages
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # Price chart
                ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
                ax1.plot(data.index, data['MA20'], label='20-Day MA', alpha=0.7)
                ax1.plot(data.index, data['MA50'], label='50-Day MA', alpha=0.7)
                ax1.set_ylabel('Price ($)')
                ax1.set_title(f'{selected_symbol} - Historical Prices with Moving Averages')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Volume chart
                ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volume')
                ax2.set_title('Trading Volume')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Volatility analysis
                data['Returns'] = data['Close'].pct_change()
                data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(data.index, data['Volatility'], label='30-Day Rolling Volatility', color='red')
                ax.set_xlabel('Date')
                ax.set_ylabel('Volatility')
                ax.set_title(f'{selected_symbol} - Volatility Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

                # Risk assessment
                current_volatility = data['Volatility'].iloc[-1]
                avg_volatility = data['Volatility'].mean()

                st.subheader('âš ï¸ Risk Assessment')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Volatility", f"{current_volatility:.2%}")
                with col2:
                    st.metric("Average Volatility", f"{avg_volatility:.2%}")

                if current_volatility > avg_volatility * 1.5:
                    st.warning("âš ï¸ High volatility detected - Higher risk!")
                elif current_volatility < avg_volatility * 0.5:
                    st.success("âœ… Low volatility - Lower risk")
                else:
                    st.info("â„¹ï¸ Normal volatility levels")

            else:
                st.error("Failed to generate predictions. Please check your data and model.")

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please try with different parameters or check your data.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><strong>Disclaimer:</strong> This is for educational purposes only. 
            Not financial advice. Always consult with a financial advisor before making investment decisions.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()