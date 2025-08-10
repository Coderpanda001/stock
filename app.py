# app.py
"""
Ultimate Stock Dashboard — Searchable + Loaded Edition
Added: Fixed timeframe dropdown + Custom From / To date pickers (custom overrides fixed)
Keeps all heavy features: indicators, intraday, backtest, model predictions, exports.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dt_time
import pytz
import os
import time
import traceback
from sklearn.preprocessing import MinMaxScaler

# optional import of keras only if user wants predictions
try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="📈 Stock Prediction", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Stock Dashboard")
st.caption("Search any Yahoo Finance ticker OR use the dropdown.")

# --------------------------
# Helper utilities
# --------------------------
def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns into single strings when needed."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                new_cols.append("_".join([str(c) for c in col if c is not None and str(c) != ""]))
            else:
                new_cols.append(str(col))
        df = df.copy()
        df.columns = new_cols
    return df

def detect_close_column_name(df: pd.DataFrame):
    """Detect the most likely Close column name in the dataframe."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lowered = [str(c).lower() for c in cols]
    for c, lc in zip(cols, lowered):
        if lc == 'close' or lc == 'adj close' or lc == 'adj_close' or lc == 'adjclose':
            return c
    for c, lc in zip(cols, lowered):
        if lc.endswith('close') or lc.startswith('close') or ('close' in lc):
            return c
    return None

def detect_volume_column_name(df: pd.DataFrame):
    """Detect the most likely Volume column name in the dataframe."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lowered = [str(c).lower() for c in cols]
    for c, lc in zip(cols, lowered):
        if lc == 'volume':
            return c
    for c, lc in zip(cols, lowered):
        if 'volume' in lc:
            return c
    return None

def ensure_datetime_index(df: pd.DataFrame):
    """Ensure the DataFrame has a DatetimeIndex; try to convert/reset if needed."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    # if index is already datetime
    if isinstance(df2.index, pd.DatetimeIndex):
        return df2
    # if there's a Date column
    if 'Date' in df2.columns:
        try:
            df2['Date'] = pd.to_datetime(df2['Date'])
            df2 = df2.set_index('Date')
            return df2
        except Exception:
            pass
    # try convert index
    try:
        df2.index = pd.to_datetime(df2.index)
        return df2
    except Exception:
        return df2

def ensure_date_column_for_plot(df: pd.DataFrame):
    """Return a DataFrame that has a 'Date' column for Plotly express (avoid using index)."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    if 'Date' in df2.columns:
        return df2
    if isinstance(df2.index, pd.DatetimeIndex):
        df2 = df2.reset_index()
        df2 = df2.rename(columns={df2.columns[0]: 'Date'})
        return df2
    # fallback: try to find date-like column
    for c in df2.columns:
        if 'date' in str(c).lower():
            df2 = df2.rename(columns={c: 'Date'})
            return df2
    return df2

def safe_fetch_history(ticker: str, start: datetime, end: datetime):
    """Fetch historical data and flatten MultiIndex columns."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = flatten_multiindex_columns(df)
        df = ensure_datetime_index(df)
        return df
    except Exception:
        return pd.DataFrame()

def compute_technical_indicators(df: pd.DataFrame):
    """Compute indicators robustly and safely avoiding math domain errors."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # detect and rename Close
    close_col = detect_close_column_name(df)
    if close_col and close_col != 'Close':
        df = df.rename(columns={close_col: 'Close'})
    if 'Close' not in df.columns:
        return df

    # ensure numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    for c in ['Open', 'High', 'Low', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # fill small gaps by forward/backfill for indicators (but don't change original too much)
    df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')

    # simple moving averages
    df['MA10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA100'] = df['Close'].rolling(100, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()

    # EMA
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger
    df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    # RSI (safe)
    eps = 1e-8
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=14, min_periods=1).mean()
    ma_down = down.rolling(window=14, min_periods=1).mean().replace(0, eps)
    rs = ma_up / ma_down
    df['RSI14'] = 100 - (100 / (1 + rs))

    # ATR (safe)
    if 'High' in df.columns and 'Low' in df.columns:
        prev_close = df['Close'].shift(1)
        tr = pd.concat([(df['High'] - df['Low']).abs(), (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14, min_periods=1).mean().fillna(method='bfill').fillna(0)
    else:
        df['ATR14'] = 0.0

    # ADX (approx safe)
    if 'High' in df.columns and 'Low' in df.columns:
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr14 = tr.rolling(14, min_periods=1).sum().replace(0, eps)
        plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / tr14)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace({0: eps}))
        df['ADX'] = dx.rolling(14, min_periods=1).mean().fillna(0)
    else:
        df['ADX'] = 0.0

    # returns and drawdown
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return'].fillna(0)).cumprod() - 1
    df['Rolling_Max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']

    return df

def candlestick_with_overlays(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.get('Open', df['Close']), high=df.get('High', df['Close']),
        low=df.get('Low', df['Close']), close=df['Close'], name='OHLC'
    ))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')))
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green')))
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(dash='dot'), opacity=0.4))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(dash='dot'), opacity=0.4))
    fig.update_layout(template='plotly_dark', title=f"{ticker} — Price with overlays", xaxis_rangeslider_visible=False, height=600)
    return fig

def simple_backtest(df, short='MA20', long='MA50'):
    df_bt = df.copy().dropna(subset=[short, long, 'Close'])
    position = 0
    entry_price = 0.0
    trades = []
    equity = 1.0
    equity_series = []
    for i, (idx, row) in enumerate(df_bt.iterrows()):
        short_v = row[short]
        long_v = row[long]
        price = row['Close']
        if short_v > long_v and position == 0:
            position = 1
            entry_price = price
            trades.append({'date': idx, 'type': 'buy', 'price': price})
        elif short_v < long_v and position == 1:
            position = 0
            ret = (price - entry_price) / entry_price if entry_price != 0 else 0
            equity = equity * (1 + ret)
            trades.append({'date': idx, 'type': 'sell', 'price': price, 'ret': ret, 'equity': equity})
        equity_series.append({'date': idx, 'equity': equity})
    eq_df = pd.DataFrame(equity_series).set_index('date') if equity_series else pd.DataFrame()
    return trades, eq_df

# --------------------------
# Sidebar: search + quick picks + fixed/custom timeframe + options
# --------------------------
with st.sidebar:
    st.header("Controls")
    st.write("Type any Yahoo ticker (e.g. AAPL, TSLA, RELIANCE.NS) then press Enter or pick from quick list.")
    search_ticker = st.text_input("Search ticker (type & press Enter)", value="")
    quick_list = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "RELIANCE.NS", "TCS.NS", "NFLX"]
    picked = st.selectbox("Or pick quick ticker", quick_list, index=0)
    # prefer search input if non-empty
    ticker = search_ticker.strip().upper() if search_ticker.strip() != "" else picked

    st.markdown("---")
    st.subheader("Timeframe")
    fixed_tf = st.selectbox("Fixed timeframe", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Max"], index=2)

    st.markdown("Or choose custom exact dates (custom overrides fixed):")
    col_from, col_to = st.columns(2)
    with col_from:
        custom_from = st.date_input("From date", value=None)
    with col_to:
        custom_to = st.date_input("To date", value=None)

    # decide effective start/end
    now = datetime.now()
    if fixed_tf == "1 Month":
        default_start = now - timedelta(days=30)
    elif fixed_tf == "3 Months":
        default_start = now - timedelta(days=90)
    elif fixed_tf == "6 Months":
        default_start = now - timedelta(days=180)
    elif fixed_tf == "1 Year":
        default_start = now - timedelta(days=365)
    elif fixed_tf == "5 Years":
        default_start = now - timedelta(days=365*5)
    else:
        default_start = now - timedelta(days=365*15)  # Max ~ 15 years

    # override fixed if both custom_from and custom_to are provided and valid
    if custom_from is not None and custom_to is not None:
        # ensure correct ordering
        if custom_from > custom_to:
            st.sidebar.error("From date must be before To date.")
            start_date = default_start
            end_date = now
        else:
            # set start/end to custom (convert to datetimes at midnight)
            start_date = datetime.combine(custom_from, dt_time.min)
            end_date = datetime.combine(custom_to, dt_time.max)
    else:
        # use fixed timeframe defaults
        start_date = datetime.combine(default_start.date(), dt_time.min)
        end_date = datetime.combine(now.date(), dt_time.max)

    st.markdown("---")
    st.subheader("Indicators & Views")
    show_ma = st.checkbox("Show Moving Averages (MA20/MA50)", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI (14)", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_adx = st.checkbox("Show ADX", value=False)
    show_volume = st.checkbox("Show Volume", value=True)
    st.markdown("---")
    st.subheader("Intraday & Live")
    intraday_toggle = st.checkbox("Enable intraday 1d chart", value=False)
    intraday_interval = st.selectbox("Intraday interval", ["1m", "2m", "5m", "15m"], index=2)
    auto_refresh = st.checkbox("Auto-refresh intraday (experimental)", value=False)
    refresh_interval = st.number_input("Refresh seconds", min_value=10, max_value=600, value=60, step=5)
    st.markdown("---")
    st.subheader("Model Prediction")
    model_path = st.text_input("Keras model path (leave empty to skip)", value=r"C:\Users\shara\OneDrive\Desktop\streamlit\stock.keras")
    run_model = st.checkbox("Run model predictions (may take time)", value=False)
    st.markdown("---")
    st.subheader("Backtest & Export")
    run_backtest = st.checkbox("Run MA crossover backtest (MA20/MA50)", value=True)
    export_csv = st.checkbox("Show export buttons", value=True)

# --------------------------
# Fetch historical data
# --------------------------
st.info(f"Fetching history for {ticker} from {start_date.date()} to {end_date.date()} ...")
raw = safe_fetch_history(ticker, start_date, end_date)
if raw is None or raw.empty:
    st.error("No historical data returned. Double-check ticker and date range.")
    st.stop()

# rename common close column to 'Close' if needed
close_col_name = detect_close_column_name(raw)
if close_col_name and close_col_name != 'Close':
    raw = raw.rename(columns={close_col_name: 'Close'})

# ensure numeric types, ensure index is datetime
raw = ensure_datetime_index(raw)
# compute indicators
hist = compute_technical_indicators(raw)

# ensure Date column for Plotly express
hist_for_plot = ensure_date_column_for_plot(hist)

# --------------------------
# Top metrics and market progress
# --------------------------
st.subheader(f"{ticker} — Overview")
latest = hist['Close'].iloc[-1]
prev = hist['Close'].iloc[-2] if len(hist) > 1 else latest
change = latest - prev
change_pct = (change / prev) * 100 if prev != 0 else 0.0

# safe 52-week range
if len(hist) >= 252:
    hi_52 = hist['Close'].tail(252).max()
    lo_52 = hist['Close'].tail(252).min()
else:
    hi_52 = hist['Close'].max()
    lo_52 = hist['Close'].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"${latest:,.2f}", f"{change_pct:.2f}%")
c2.metric("Today Change", f"${change:,.2f}", f"{change_pct:.2f}%")
c3.metric("30d Volatility (std)", f"{hist['Daily_Return'].tail(30).std():.4f}")
c4.metric("52-week Range", f"${lo_52:,.2f} — ${hi_52:,.2f}")

# market progress (NY)
def market_progress():
    ny = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny)
    open_t = ny.localize(datetime.combine(now_ny.date(), dt_time(9,30)))
    close_t = ny.localize(datetime.combine(now_ny.date(), dt_time(16,0)))
    if now_ny < open_t:
        return 0, "Pre-market"
    if now_ny > close_t:
        return 100, "After-hours"
    pct = int(((now_ny - open_t).total_seconds() / (close_t - open_t).total_seconds()) * 100)
    return pct, "Open"

pct, state = market_progress()
st.write(f"Market status: **{state}**")
st.progress(pct)

# --------------------------
# Price & overlays chart
# --------------------------
st.markdown("### Price Chart & Overlays")
fig_main = candlestick_with_overlays(hist, ticker)
st.plotly_chart(fig_main, use_container_width=True)

# --------------------------
# Quick stats panel
# --------------------------
st.subheader("Quick Stats")
qs1, qs2 = st.columns(2)
with qs1:
    st.write(f"**Latest:** {latest:,.2f}")
    st.write(f"**Change:** {change:,.2f} ({change_pct:.2f}%)")
    if 'RSI14' in hist.columns:
        st.write(f"**RSI(14):** {hist['RSI14'].iloc[-1]:.2f}")
with qs2:
    if 'MACD' in hist.columns:
        st.write(f"**MACD:** {hist['MACD'].iloc[-1]:.4f}")
    if 'ADX' in hist.columns:
        st.write(f"**ADX:** {hist['ADX'].iloc[-1]:.2f}")
    st.write(f"**Data points:** {len(hist)}")

# --------------------------
# Intraday (optional)
# --------------------------
st.markdown("### Intraday (1d) — optional")
if intraday_toggle:
    try:
        intr = yf.download(ticker, period="1d", interval=intraday_interval, progress=False, threads=False)
        if intr is None or intr.empty:
            st.info("No intraday data available for this ticker.")
        else:
            intr = flatten_multiindex_columns(intr)
            intr = ensure_datetime_index(intr)
            # detect close col
            close_intr = detect_close_column_name(intr)
            if close_intr and close_intr != 'Close':
                intr = intr.rename(columns={close_intr: 'Close'})
            # plot intraday
            fig_intr = go.Figure()
            fig_intr.add_trace(go.Scatter(x=intr.index, y=intr['Close'], name='Intraday Close'))
            fig_intr.update_layout(title=f"Intraday {ticker} ({intraday_interval})", template='plotly_dark', height=320)
            st.plotly_chart(fig_intr, use_container_width=True)
            # volume heatmap
            vol_name_intr = detect_volume_column_name(intr)
            if vol_name_intr:
                intr['hour'] = intr.index.hour
                intr['dow'] = intr.index.day_name()
                heat = intr.pivot_table(values=vol_name_intr, index='hour', columns='dow', aggfunc='sum').fillna(0)
                fig_heat = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='Viridis'))
                fig_heat.update_layout(title='Intraday Volume Heatmap (hour vs day)', template='plotly_dark')
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Intraday volume not available.")
    except Exception as e:
        st.info("Intraday fetch/plot failed: " + str(e))

    # auto-refresh experimental
    if auto_refresh:
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed > refresh_interval:
            st.session_state.last_refresh = time.time()
            st.experimental_rerun()

# --------------------------
# Volume plotting (robust)
# --------------------------
st.markdown("### Volume")
vol_candidate = detect_volume_column_name(hist)
if vol_candidate is not None and show_volume:
    hist_plot = ensure_date_column_for_plot(hist)
    if 'Date' not in hist_plot.columns:
        hist_plot['Date'] = hist_plot.index
    try:
        fig_vol = px.bar(hist_plot, x='Date', y=vol_candidate, title='Daily Volume', template='plotly_dark')
        fig_vol.update_layout(height=300)
        st.plotly_chart(fig_vol, use_container_width=True)
    except Exception as e:
        st.info("Volume plot failed: " + str(e))
else:
    st.info("Volume display disabled or volume data missing.")

# --------------------------
# Returns, cumulative, drawdown
# --------------------------
st.markdown("### Returns & Risk")
rcol1, rcol2 = st.columns(2)
with rcol1:
    if 'Daily_Return' in hist.columns:
        dfr = hist.reset_index()
        fig_hist = px.histogram(dfr, x='Daily_Return', nbins=80, title='Daily Returns Distribution', template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Daily returns not available.")
with rcol2:
    if 'Cumulative_Return' in hist.columns:
        dfc = ensure_date_column_for_plot(hist)
        if 'Date' not in dfc.columns:
            dfc['Date'] = dfc.index
        fig_cum = px.line(dfc, x='Date', y='Cumulative_Return', title='Cumulative Return', template='plotly_dark')
        st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("### Drawdown")
if 'Drawdown' in hist.columns:
    ddf = ensure_date_column_for_plot(hist)
    if 'Date' not in ddf.columns:
        ddf['Date'] = ddf.index
    fig_dd = px.area(ddf, x='Date', y='Drawdown', title='Drawdown (negative is drawdown)', template='plotly_dark')
    st.plotly_chart(fig_dd, use_container_width=True)
else:
    st.info("Drawdown not available.")

# --------------------------
# MACD & RSI
# --------------------------
st.markdown("### MACD & RSI")
mcol, scol = st.columns(2)
with mcol:
    if 'MACD' in hist.columns:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'))
        fig_m.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal'))
        if 'MACD_Hist' in hist.columns:
            fig_m.add_bar(x=hist.index, y=hist['MACD_Hist'], name='Hist')
        fig_m.update_layout(title='MACD', template='plotly_dark', height=300)
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("MACD not available.")
with scol:
    if 'RSI14' in hist.columns:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=hist.index, y=hist['RSI14'], name='RSI'))
        fig_r.add_hline(y=70, line_dash='dash', line_color='red')
        fig_r.add_hline(y=30, line_dash='dash', line_color='green')
        fig_r.update_layout(title='RSI (14)', template='plotly_dark', height=300)
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("RSI not available.")

# --------------------------
# ADX / ATR (optional)
# --------------------------
st.markdown("### ADX & ATR")
a1, a2 = st.columns(2)
with a1:
    if 'ADX' in hist.columns and show_adx:
        adf = ensure_date_column_for_plot(hist)
        if 'Date' not in adf.columns:
            adf['Date'] = adf.index
        fig_adx = px.line(adf, x='Date', y='ADX', title='ADX (trend strength)', template='plotly_dark')
        st.plotly_chart(fig_adx, use_container_width=True)
    else:
        if show_adx:
            st.info("ADX not available.")
with a2:
    if 'ATR14' in hist.columns:
        adf2 = ensure_date_column_for_plot(hist)
        if 'Date' not in adf2.columns:
            adf2['Date'] = adf2.index
        fig_atr = px.line(adf2, x='Date', y='ATR14', title='ATR (14)', template='plotly_dark')
        st.plotly_chart(fig_atr, use_container_width=True)
    else:
        st.info("ATR not available.")

# --------------------------
# Backtest (MA20/MA50)
# --------------------------
st.markdown("### Backtest: MA20 vs MA50 (simple)")
if run_backtest:
    try:
        trades, equity = simple_backtest(hist, short='MA20', long='MA50')
        if trades:
            st.write("Recent trades (last 10):")
            st.dataframe(pd.DataFrame(trades).tail(10))
        else:
            st.write("No trades from simple MA crossover in selected timeframe.")
        if not equity.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=equity.index, y=equity['equity'], name='Equity'))
            fig_eq.update_layout(title='Strategy Equity Curve', template='plotly_dark')
            st.plotly_chart(fig_eq, use_container_width=True)
    except Exception as e:
        st.info("Backtest failed: " + str(e))
else:
    st.info("Backtest disabled.")

# --------------------------
# Optional model prediction (robust)
# --------------------------
st.markdown("### Model Predictions (optional & defensive)")
preds_df = None
if run_model and model_path.strip() != "":
    if os.path.exists(model_path) and KERAS_AVAILABLE:
        try:
            st.info("Loading model (may take time)...")
            model = load_model(model_path)
            st.success("Model loaded.")
            lookback = 100
            close_series = hist['Close'].dropna().astype(float)
            # clean
            close_series = close_series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(close_series) <= lookback + 5:
                st.warning("Not enough data for model windows. Increase timeframe.")
            else:
                split_idx = int(len(close_series) * 0.6)
                test_chunk = close_series.iloc[split_idx:].to_frame(name='Close')
                test_chunk = test_chunk.replace([np.inf, -np.inf], np.nan).dropna()
                if len(test_chunk) <= lookback + 1:
                    st.warning("Not enough valid rows after cleaning.")
                else:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled = scaler.fit_transform(test_chunk[['Close']].values.reshape(-1, 1))
                    X = []
                    for i in range(lookback, len(scaled)):
                        X.append(scaled[i - lookback:i, 0])
                    X = np.array(X).reshape(-1, lookback, 1)
                    if not np.isfinite(X).all():
                        st.error("Model input contains NaN/Inf. Aborting predictions.")
                    else:
                        try:
                            preds = model.predict(X)
                            preds = np.asarray(preds)
                            try:
                                preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                            except Exception:
                                min_v = test_chunk['Close'].min()
                                max_v = test_chunk['Close'].max()
                                preds_inv = preds.flatten() * (max_v - min_v) + min_v
                            pred_index = test_chunk.index[lookback:]
                            preds_df = pd.DataFrame({'Actual': test_chunk['Close'].iloc[lookback:].values, 'Predicted': preds_inv}, index=pred_index)
                            st.success("Predictions ready.")
                            st.dataframe(preds_df.tail(20).round(2))
                            fig_p = go.Figure()
                            fig_p.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Actual'], name='Actual'))
                            fig_p.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Predicted'], name='Predicted'))
                            fig_p.update_layout(title='Model: Actual vs Predicted', template='plotly_dark')
                            st.plotly_chart(fig_p, use_container_width=True)
                        except Exception:
                            st.error("Model prediction failed. See console for traceback.")
                            st.text(traceback.format_exc())
        except Exception:
            st.error("Failed to load model. See traceback.")
            st.text(traceback.format_exc())
    else:
        if not KERAS_AVAILABLE:
            st.warning("TensorFlow/Keras not available in environment; cannot run model.")
        else:
            st.warning("Provided model path does not exist; skipping predictions.")
else:
    st.info("Model predictions skipped (unchecked or path empty).")

# --------------------------
# Alerts (RSI/MACD cross)
# --------------------------
st.markdown("### Alerts & Signals")
alerts = []
if 'RSI14' in hist.columns and show_rsi:
    rsi_now = hist['RSI14'].iloc[-1]
    if rsi_now > 70:
        alerts.append(f"RSI(14) = {rsi_now:.1f} → Overbought (caution).")
    elif rsi_now < 30:
        alerts.append(f"RSI(14) = {rsi_now:.1f} → Oversold.")

if 'MACD' in hist.columns and 'MACD_Signal' in hist.columns and show_macd:
    if len(hist) >= 3:
        m = hist['MACD'].iloc[-3:]
        s = hist['MACD_Signal'].iloc[-3:]
        if (m.iloc[-2] < s.iloc[-2]) and (m.iloc[-1] > s.iloc[-1]):
            alerts.append("MACD bullish crossover detected.")
        if (m.iloc[-2] > s.iloc[-2]) and (m.iloc[-1] < s.iloc[-1]):
            alerts.append("MACD bearish crossover detected.")

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.info("No alerts at this time (RSI/MACD).")

# --------------------------
# Peer comparison & correlation
# --------------------------
st.markdown("### Peer Comparison & Correlation")
peer_input = st.text_input("Peer tickers (comma-separated)", value="")
if peer_input.strip():
    peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]
    closes = {ticker: hist['Close']}
    for p in peers:
        dfp = safe_fetch_history(p, start_date, end_date)
        if not dfp.empty:
            close_p = detect_close_column_name(dfp)
            if close_p and close_p != 'Close':
                dfp = dfp.rename(columns={close_p: 'Close'})
            dfp = ensure_datetime_index(dfp)
            if 'Close' in dfp.columns:
                closes[p] = dfp['Close']
    if len(closes) > 1:
        merged = pd.concat(closes, axis=1).dropna()
        normalized = merged / merged.iloc[0] * 100
        fig_cmp = go.Figure()
        for col in normalized.columns:
            fig_cmp.add_trace(go.Scatter(x=normalized.index, y=normalized[col], name=col))
        fig_cmp.update_layout(title='Normalized price comparison (base=100)', template='plotly_dark')
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Not enough peer data for comparison.")

# --------------------------
# Exports
# --------------------------
st.markdown("### Exports")
if export_csv:
    try:
        csv_hist = hist.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button("Download historical CSV", csv_hist, f"{ticker}_history.csv", "text/csv")
    except Exception as e:
        st.info("Export failed: " + str(e))
if preds_df is not None:
    try:
        csv_pred = preds_df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv_pred, f"{ticker}_predictions.csv", "text/csv")
    except Exception as e:
        st.info("Prediction export failed: " + str(e))

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("""
**Notes**
- Intraday availability depends on Yahoo Finance and the symbol; some tickers / regions may not support 1m/2m intervals.
- Model predictions require a Keras/TensorFlow model compatible with your preprocessing.
- Backtest is simplistic — ignores fees, slippage, and other real-world constraints.
- If anything crashes, copy the console traceback (or paste the Streamlit printed error) and I'll patch it.
""")
