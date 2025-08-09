import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import time

# Check if tensorflow/keras is available
try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.keras.models import load_model
        KERAS_AVAILABLE = True
    except ImportError:
        KERAS_AVAILABLE = False
        st.error("❌ Keras/TensorFlow not found. Please add 'tensorflow>=2.13.0' to requirements.txt")

@st.cache_data(ttl=600)  # Cache for 10 minutes
def download_stock_data_safe(symbol, start, end, max_retries=2):
    """Safely download stock data with error handling"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(1)  # Brief delay between retries
            
            data = yf.download(
                symbol, 
                start=start, 
                end=end, 
                progress=False,
                timeout=15
            )
            
            if not data.empty:
                return data, None
                
        except Exception as e:
            if attempt == max_retries - 1:
                return None, str(e)
    
    return None, "No data available"

def suggest_action(predicted_price, current_price):
    """Generate trading suggestion"""
    change = ((predicted_price - current_price) / current_price) * 100
    
    if change > 2:
        return "Strong Buy", change
    elif change > 0:
        return "Buy", change  
    elif change < -2:
        return "Strong Sell", change
    else:
        return "Hold", change

def main():
    st.set_page_config(page_title="Stock Dashboard", page_icon="📈")
    st.title("📈 Stock Market Dashboard")
    
    # Check system requirements
    if not KERAS_AVAILABLE:
        st.error("⚠️ **Deployment Issue**: Machine learning model cannot be loaded.")
        st.info("This appears to be a missing dependency issue. Please check your requirements.txt file.")
        return
    
    # Sidebar inputs
    st.sidebar.header("Settings")
    
    # Stock selection
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    selected_stock = st.sidebar.selectbox("Select Stock", stocks)
    
    # Date range
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2022-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-12-31'))
    
    if start_date >= end_date:
        st.error("End date must be after start date")
        return
    
    # Download data
    with st.spinner(f'Loading {selected_stock} data...'):
        data, error = download_stock_data_safe(selected_stock, start_date, end_date)
    
    if data is None:
        st.error(f"❌ Could not load data for {selected_stock}")
        st.write(f"Error: {error}")
        
        # Troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.write("""
            **Common fixes:**
            1. Try a different stock symbol
            2. Adjust the date range
            3. Check internet connection
            4. Wait a moment and refresh
            """)
        return
    
    if len(data) < 100:
        st.warning("⚠️ Limited data available. Results may be less reliable.")
    
    # Display basic info
    st.success(f"✅ Loaded {len(data)} data points for {selected_stock}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${data.Close.iloc[-1]:.2f}")
    with col2:
        st.metric("Period High", f"${data.High.max():.2f}")
    with col3:
        st.metric("Period Low", f"${data.Low.min():.2f}")
    
    # Show recent data
    st.subheader("Recent Data")
    st.dataframe(data.tail(10))
    
    # Basic analysis without ML model for now
    st.subheader("📊 Price Analysis")
    
    # Calculate moving averages
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', alpha=0.8)
    ax.plot(data.index, data['MA20'], label='20-Day MA', alpha=0.7)
    ax.plot(data.index, data['MA50'], label='50-Day MA', alpha=0.7)
    ax.set_title(f'{selected_stock} Price History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Simple trend analysis
    recent_price = data.Close.iloc[-1]
    ma20_current = data['MA20'].iloc[-1]
    ma50_current = data['MA50'].iloc[-1]
    
    st.subheader("📈 Technical Analysis")
    
    # Generate simple signals
    if recent_price > ma20_current > ma50_current:
        signal = "Bullish Trend"
        color = "success"
    elif recent_price < ma20_current < ma50_current:
        signal = "Bearish Trend"
        color = "error"
    else:
        signal = "Mixed Signals"
        color = "warning"
    
    if color == "success":
        st.success(f"✅ {signal}")
    elif color == "error":
        st.error(f"🔻 {signal}")
    else:
        st.warning(f"⚠️ {signal}")
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    recent_volume = data['Volume'].iloc[-1]
    
    volume_ratio = recent_volume / avg_volume
    st.metric("Volume vs 20-Day Average", f"{volume_ratio:.2f}x")
    
    # Disclaimer
    st.markdown("---")
    st.info("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()
