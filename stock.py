import os
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from tensorflow.keras.models import load_model
except Exception:
    # If TF not installed, we'll still allow demo mode
    load_model = None

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

# ---------------------------
# Utility / fallback classes
# ---------------------------
class FallbackModel:
    """Very small fallback predictor: predicts next scaled value as window mean with slight drift."""
    def predict(self, x):
        # expects x shape (samples, window_size, 1)
        window_means = x.mean(axis=1).reshape(-1, 1)
        return window_means * 1.0005

@st.cache_data(ttl=600)
def download_data(symbol, start, end):
    """Download via yfinance and return DataFrame. Returns None on failure."""
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def build_windows(scaled_series, window_size):
    x, y = [], []
    for i in range(window_size, scaled_series.shape[0]):
        x.append(scaled_series[i - window_size:i].reshape(window_size, 1))
        y.append(scaled_series[i, 0])
    return np.array(x), np.array(y)

def safe_inverse(scaler, arr):
    arr2 = np.array(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr2).flatten()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📈 Stock Market Dashboard (robust demo)")
st.markdown("---")

# Options
col1, col2 = st.columns([2, 1])
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2012-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2022-12-31"))
    symbol = st.selectbox("Select Stock", ["AAPL", "GOOG", "MSFT", "AMZN"])
    use_demo = st.checkbox("Demo mode (use synthetic data if download or model fails)", value=False)

with col2:
    uploaded_model = st.file_uploader("Upload your Keras model (.keras / .h5) (optional)", type=['keras','h5','keras'])
    window_size_user = st.slider("Window size (override)", min_value=5, max_value=300, value=60)

# Load data
with st.spinner("Getting historical data..."):
    data = download_data(symbol, start_date, end_date)
    if data is None:
        if use_demo:
            st.warning("Couldn't download data; using synthetic demo data.")
            n = 400
            rng = np.random.default_rng(123)
            prices = 100 + np.cumsum(rng.normal(loc=0.2, scale=1.0, size=n))
            dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
            data = pd.DataFrame({"Close": prices}, index=dates)
        else:
            st.error("Could not download data. Turn on demo mode or check network.")
            st.stop()

# show basic data
st.subheader("Sample data")
st.dataframe(data.head())

# Try loading model (uploaded or local path)
model_obj = None
model_window_size = None
if uploaded_model is not None and load_model is not None:
    # save to temp and load
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
    tmp.write(uploaded_model.getbuffer())
    tmp.flush()
    tmp.close()
    try:
        model_obj = load_model(tmp.name)
        st.success("Model loaded from upload.")
    except Exception as e:
        st.warning(f"Uploaded model could not be loaded: {e}")
        model_obj = None
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
elif os.path.exists("stockpredict.keras") and load_model is not None:
    try:
        model_obj = load_model("stockpredict.keras")
        st.success("Model loaded from disk (stockpredict.keras).")
    except Exception as e:
        st.warning(f"Failed to load stockpredict.keras: {e}")
        model_obj = None
else:
    if load_model is None:
        st.info("TensorFlow/Keras not available in this environment; using fallback predictor if needed.")

# Determine window size
if model_obj is not None:
    try:
        # handle different possible shapes: (None, window_size, features)
        inp_shape = model_obj.input_shape
        if isinstance(inp_shape, tuple) and len(inp_shape) >= 2:
            model_window_size = int(inp_shape[1])
    except Exception:
        model_window_size = None

window_size = model_window_size or window_size_user

if len(data) < window_size + 1:
    st.error(f"Not enough data for window_size={window_size}. Need at least {window_size+1} rows.")
    st.stop()

# Prepare train/test
data_train = pd.DataFrame(data['Close'].iloc[: int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'].iloc[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train[['Close']])

# prefix test with last window_size from train
past_days = data_train.tail(window_size)
data_test_full = pd.concat([past_days, data_test], ignore_index=True)

data_test_scale = scaler.transform(data_test_full[['Close']])

# Build windows
x, y = build_windows(data_test_scale, window_size)

if x.size == 0:
    st.error("After windowing, no samples were created. Check window size vs data length.")
    st.stop()

# choose model to predict
predictor = None
if model_obj is not None:
    predictor = model_obj
else:
    predictor = FallbackModel()
    st.info("Using fallback predictor (demo).")

# Predict
try:
    pred_scaled = predictor.predict(x)  # predictor must return shape (n_samples, 1) or (n_samples,)
    pred = safe_inverse(scaler, pred_scaled)
    y_inv = safe_inverse(scaler, y)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Suggested action using last predicted vs last close
last_pred = float(pred[-1])
last_close = float(data_test_full['Close'].iloc[-1])
suggested_action = "Buy" if last_pred > last_close else "Sell"

# Output: plots + tables
st.subheader("🔍 Predicted vs Actual (test portion)")
results = pd.DataFrame({'Predicted': pred, 'Actual': y_inv})
st.dataframe(results.tail(10))

fig, ax = plt.subplots()
ax.plot(results['Actual'], label='Actual')
ax.plot(results['Predicted'], label='Predicted')
ax.set_xlabel("Samples")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

st.subheader("📜 Historical Closing Prices")
fig2, ax2 = plt.subplots()
ax2.plot(data['Close'], label='Closing Price')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.subheader("✅ Suggested Action")
if suggested_action == "Buy":
    st.success(f"Suggested Action: {suggested_action} — Predicted {last_pred:.2f} vs Current {last_close:.2f}")
else:
    st.error(f"Suggested Action: {suggested_action} — Predicted {last_pred:.2f} vs Current {last_close:.2f}")

st.markdown("---")
st.markdown("Contact / Support: https://tradelitcare.streamlit.app")