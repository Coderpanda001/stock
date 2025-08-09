import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Cache data downloads to improve performance
@st.cache_data
def download_stock_data(symbol, start, end):
    """Download stock data with caching to avoid repeated API calls"""
    return yf.download(symbol, start=start, end=end, progress=False)

# Load the pre-trained model with error handling
@st.cache_resource
def load_prediction_model():
    """Load the pre-trained model with error handling"""
    try:
        return load_model("stockpredict.keras")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_prediction_model()

def predict_and_suggest_action(data_test_scale, scaler, window_size):
    """Generate predictions using the trained model"""
    x = []
    y = []

    for i in range(window_size, data_test_scale.shape[0]):
        x.append(data_test_scale[i - window_size:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    if len(x) == 0:
        raise ValueError("Not enough data for prediction")

    predict = model.predict(x)

    # Inverse transform
    predict = scaler.inverse_transform(predict)
    y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    return predict.flatten(), y

def suggest_action(predicted_price, current_price):
    """Suggest buy/sell action based on prediction"""
    if predicted_price > current_price * 1.02:  # 2% threshold for buy
        return "Strong Buy"
    elif predicted_price > current_price:
        return "Buy"
    elif predicted_price < current_price * 0.98:  # 2% threshold for sell
        return "Strong Sell"
    else:
        return "Hold"

def validate_data(data, min_data_points=100):
    """Validate downloaded data"""
    if data is None or data.empty:
        return False, "No data available"
    
    if len(data) < min_data_points:
        return False, f"Insufficient data: Only {len(data)} data points available. Need at least {min_data_points}."
    
    return True, "Data is valid"

def main():
    st.set_page_config(
        page_title="Stock Market Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title('📈 Stock Market Dashboard')
    st.markdown("---")

    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Configuration")
        
        # Date range with validation
        st.subheader("📅 Select Data Range")
        
        # Default dates
        default_start = pd.to_datetime('2020-01-01')
        default_end = pd.to_datetime('2023-12-31')
        
        start_date = st.date_input(
            "Start Date", 
            value=default_start,
            min_value=pd.to_datetime('2010-01-01'),
            max_value=pd.to_datetime('2024-12-31')
        )
        
        end_date = st.date_input(
            "End Date", 
            value=default_end,
            min_value=pd.to_datetime('2010-01-01'),
            max_value=pd.to_datetime('2024-12-31')
        )

        # Date validation
        if start_date >= end_date:
            st.error("❌ End date must be after start date.")
            st.stop()

        # Check date range duration
        date_diff = (end_date - start_date).days
        if date_diff < 365:
            st.warning("⚠️ Date range is less than 1 year. Consider selecting a longer period for better predictions.")

        # Convert to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Stock selection
        st.subheader("🏢 Select Stock")
        stock_options = {
            'AAPL': 'Apple Inc.',
            'GOOG': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corp.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corp.',
            'NFLX': 'Netflix Inc.'
        }
        
        selected_stock = st.selectbox(
            'Choose Stock Symbol', 
            options=list(stock_options.keys()),
            format_func=lambda x: f"{x} - {stock_options[x]}"
        )

        # Train/Test split ratio
        st.subheader("🔄 Model Settings")
        train_ratio = st.slider("Training Data Ratio", 0.6, 0.9, 0.8, 0.05)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📋 Summary")
        st.write(f"**Stock:** {selected_stock} - {stock_options[selected_stock]}")
        st.write(f"**Period:** {start_date.date()} to {end_date.date()}")
        st.write(f"**Duration:** {date_diff} days")
        st.write(f"**Train Ratio:** {train_ratio:.0%}")

    with col1:
        # Download data with comprehensive error handling
        try:
            with st.spinner(f'📥 Downloading data for {selected_stock}...'):
                data = download_stock_data(selected_stock, start=start_date, end=end_date)
                
            # Validate data
            is_valid, message = validate_data(data)
            if not is_valid:
                st.error(f"❌ Data Error: {message}")
                st.write("**Possible solutions:**")
                st.write("- Try a different date range")
                st.write("- Select a different stock symbol")
                st.write("- Check if the stock was trading during this period")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            st.write("**Troubleshooting tips:**")
            st.write("- Check your internet connection")
            st.write("- Try refreshing the page")
            st.write("- Verify the stock symbol exists")
            st.stop()

        # Success message
        st.success(f"✅ Successfully downloaded {len(data)} data points for {selected_stock}")

    # Display data overview
    st.subheader("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Points", len(data))
    with col2:
        st.metric("Latest Price", f"${data.Close.iloc[-1]:.2f}")
    with col3:
        st.metric("Period High", f"${data.Close.max():.2f}")
    with col4:
        st.metric("Period Low", f"${data.Close.min():.2f}")

    # Show recent data
    with st.expander("📈 Recent Data (Last 10 days)"):
        st.dataframe(data.tail(10))

    try:
        # Prepare data for model
        split_point = int(len(data) * train_ratio)
        data_train = pd.DataFrame(data.Close[:split_point])
        data_test = pd.DataFrame(data.Close[split_point:])

        # Check if we have enough test data
        if len(data_test) < 20:
            st.error("❌ Not enough test data. Please:")
            st.write(f"- Current test data points: {len(data_test)}")
            st.write("- Try selecting a longer date range")
            st.write("- Or reduce the training ratio")
            st.stop()

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_train)

        # Get model's input window size
        window_size = model.input_shape[1]
        
        # Prepare test data with proper window
        past_days = data_train.tail(window_size)
        data_test_combined = pd.concat([past_days, data_test], ignore_index=True)
        data_test_scale = scaler.transform(data_test_combined)

        # Make predictions
        with st.spinner('🔮 Generating predictions...'):
            predict, y = predict_and_suggest_action(data_test_scale, scaler, window_size)

        # Calculate accuracy metrics
        mape = np.mean(np.abs((y - predict) / y)) * 100
        rmse = np.sqrt(np.mean((y - predict) ** 2))

        # Suggest action
        current_price = float(data.Close.iloc[-1])
        predicted_price = float(predict[-1])
        suggested_action = suggest_action(predicted_price, current_price)
        price_change_pct = ((predicted_price - current_price) / current_price) * 100

        # Display prediction results
        st.markdown("---")
        st.subheader('🔍 Prediction Results')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}", f"{price_change_pct:+.2f}%")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        with col4:
            st.metric("RMSE", f"${rmse:.2f}")

        # Action suggestion with styling
        st.subheader('💡 Suggested Action')
        if "Buy" in suggested_action:
            st.success(f"✅ **{suggested_action}**")
            st.write(f"The model predicts a {price_change_pct:+.2f}% price movement.")
        elif "Sell" in suggested_action:
            st.error(f"🔻 **{suggested_action}**")
            st.write(f"The model predicts a {price_change_pct:+.2f}% price movement.")
        else:
            st.info(f"⏸️ **{suggested_action}**")
            st.write(f"The model predicts a {price_change_pct:+.2f}% price movement.")

        # Visualization section
        st.markdown("---")
        st.subheader('📈 Analysis & Visualizations')
        
        # Prediction vs Actual
        st.subheader('🎯 Predicted vs Actual Prices')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y, 'r-', label='Actual Price', alpha=0.8)
        ax.plot(predict, 'g--', label='Predicted Price', alpha=0.8)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{selected_stock} - Prediction vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Historical prices with moving averages
        st.subheader('📊 Historical Prices & Moving Averages')
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], label='Closing Price', alpha=0.8)
        ax.plot(data.index, data['MA20'], label='20-Day MA', alpha=0.7)
        ax.plot(data.index, data['MA50'], label='50-Day MA', alpha=0.7)
        ax.plot(data.index, data['MA200'], label='200-Day MA', alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{selected_stock} - Price History & Moving Averages')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Volatility analysis
        st.subheader('📉 Volatility Analysis')
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)  # Annualized
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Returns
        ax1.plot(data.index, data['Returns'], alpha=0.6, color='blue')
        ax1.set_ylabel('Daily Returns')
        ax1.set_title('Daily Returns')
        ax1.grid(True, alpha=0.3)
        
        # Volatility
        ax2.plot(data.index, data['Volatility'], color='red', alpha=0.8)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Annualized Volatility')
        ax2.set_title('Rolling 30-Day Volatility (Annualized)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Detailed results table
        with st.expander("📋 Detailed Prediction Results"):
            results_df = pd.DataFrame({
                'Actual Price': y,
                'Predicted Price': predict,
                'Absolute Error': np.abs(y - predict),
                'Percentage Error': np.abs((y - predict) / y) * 100
            })
            st.dataframe(results_df.round(4))

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.write("**Possible causes:**")
        st.write("- Model file is corrupted or incompatible")
        st.write("- Insufficient data for prediction")
        st.write("- Data preprocessing error")

    # Footer
    st.markdown("---")
    st.markdown("### 📞 Support & Contact")
    st.markdown("**Need help?** Visit our support page: [TradeCareLite Support](https://tradelitcare.streamlit.app)")
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer"):
        st.markdown("""
        **Investment Disclaimer:**
        - This tool is for educational and informational purposes only
        - Past performance does not guarantee future results
        - All investments carry risk of loss
        - Consult with a qualified financial advisor before making investment decisions
        - The creators are not responsible for any financial losses
        
        **Data Sources:**
        - Intraday data provided by FACTSET and subject to terms of use
        - Historical and current end-of-day data provided by FACTSET
        - All quotes are in local exchange time
        - Real-time data may be delayed at least 15 minutes
        """)

if __name__ == "__main__":
    main()
