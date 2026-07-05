# ============================================================
# Tactical ETF Rotation Strategy Dashboard
# Created by Zaeem Al Salami
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ETF Rotation Dashboard | Zaeem Al Salami",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #A9B4C2;
        margin-top: 0px;
    }

    .name {
        font-size: 18px;
        color: #5DADE2;
        font-weight: 600;
    }

    .section-header {
        font-size: 26px;
        color: #FFFFFF;
        font-weight: 700;
        margin-top: 30px;
    }

    .metric-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        text-align: center;
    }

    .metric-label {
        color: #A9B4C2;
        font-size: 14px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="title">Tactical ETF Rotation Strategy</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Python-based investment allocation and risk dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="name">Created by Zaeem Al Salami</p>', unsafe_allow_html=True)

st.write("---")


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title("Dashboard Settings")

tickers = st.sidebar.multiselect(
    "Select ETF universe",
    ["SPY", "QQQ", "TLT", "AGG", "GLD", "EEM", "BIL"],
    default=["SPY", "QQQ", "TLT", "AGG", "GLD", "EEM", "BIL"]
)

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2010-01-01"))

top_n = st.sidebar.slider("Number of ETFs selected", 1, 5, 3)

transaction_cost_bps = st.sidebar.slider(
    "Transaction cost in basis points",
    0,
    50,
    5
)

initial_capital = st.sidebar.number_input(
    "Initial capital",
    min_value=1000,
    value=100000,
    step=10000
)


# ============================================================
# DATA DOWNLOAD
# ============================================================

@st.cache_data
def download_prices(tickers, start_date):
    """
    Downloads ETF prices from Yahoo Finance.
    Uses adjusted prices through yfinance auto_adjust=True.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No data downloaded. Please check tickers or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.ffill().dropna()
    return prices


try:
    prices = download_prices(tickers, start_date)
except Exception as e:
    st.error(f"Data download failed: {e}")
    st.stop()


# ============================================================
# RETURN CALCULATIONS
# ============================================================

daily_returns = prices.pct_change().dropna()
monthly_prices = prices.resample("M").last()
monthly_returns = monthly_prices.pct_change().dropna()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

mom_3m = monthly_prices.pct_change(3)
mom_6m = monthly_prices.pct_change(6)
mom_12m = monthly_prices.pct_change(12)

weighted_momentum = (0.30 * mom_3m) + (0.30 * mom_6m) + (0.40 * mom_12m)

daily_volatility = daily_returns.rolling(63).std() * np.sqrt(252)
monthly_volatility = daily_volatility.resample("M").last()

moving_average_200 = prices.rolling(200).mean()
trend_signal = (prices > moving_average_200).resample("M").last()

risk_adjusted_score = weighted_momentum / monthly_volatility
risk_adjusted_score = risk_adjusted_score.replace([np.inf, -np.inf], np.nan)


# ============================================================
# BUILD TARGET WEIGHTS
# ============================================================

def build_target_weights(scores, trend, tickers, top_n):
    """
    Selects top ETFs using positive risk-adjusted momentum and trend filter.
    If no ETF qualifies, allocates to BIL when available.
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=tickers)

    cash_ticker = "BIL" if "BIL" in tickers else tickers[0]
    risk_assets = [ticker for ticker in tickers if ticker != cash_ticker]

    for date in scores.index:
        if date not in trend.index:
            continue

        date_scores = scores.loc[date, risk_assets].dropna()
        date_trend = trend.loc[date, risk_assets].reindex(date_scores.index).fillna(False)

        eligible = date_scores[(date_scores > 0) & (date_trend == True)]

        if eligible.empty:
            weights.loc[date, cash_ticker] = 1.0
        else:
            selected = eligible.sort_values(ascending=False).head(top_n).index
            weights.loc[date, selected] = 1 / len(selected)

    weights = weights.dropna(how="all")
    return weights


target_weights = build_target_weights(
    risk_adjusted_score,
    trend_signal,
    tickers,
    top_n
)


# ============================================================
# BACKTEST ENGINE
# ============================================================

common_index = monthly_returns.index.intersection(target_weights.index)
common_columns = monthly_returns.columns.intersection(target_weights.columns)

monthly_returns_aligned = monthly_returns.loc[common_index, common_columns]
target_weights_aligned = target_weights.loc[common_index, common_columns]

active_weights = target_weights_aligned.shift(1).dropna()

monthly_returns_aligned = monthly_returns_aligned.loc[active_weights.index]

gross_returns = (active_weights * monthly_returns_aligned).sum(axis=1)

turnover = target_weights_aligned.diff().abs().sum(axis=1).shift(1)
turnover = turnover.reindex(gross_returns.index).fillna(0)

transaction_cost = turnover * (transaction_cost_bps / 10000)

strategy_returns = gross_returns - transaction_cost
strategy_value = initial_capital * (1 + strategy_returns).cumprod()


# ============================================================
# BENCHMARKS
# ============================================================

benchmark_returns = pd.DataFrame(index=strategy_returns.index)

if "SPY" in monthly_returns.columns:
    benchmark_returns["SPY Buy & Hold"] = monthly_returns["SPY"].reindex(strategy_returns.index)

if "AGG" in monthly_returns.columns and "SPY" in monthly_returns.columns:
    benchmark_returns["60/40 Portfolio"] = (
        0.60 * monthly_returns["SPY"].reindex(strategy_returns.index)
        + 0.40 * monthly_returns["AGG"].reindex(strategy_returns.index)
    )

all_returns = pd.concat(
    [strategy_returns.rename("ETF Rotation Strategy"), benchmark_returns],
    axis=1
).dropna()

cumulative_returns = (1 + all_returns).cumprod()


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_drawdown(returns):
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def calculate_metrics(returns):
    returns = returns.dropna()
    months = len(returns)

    if months == 0:
        return {}

    cagr = (1 + returns).prod() ** (12 / months) - 1
    volatility = returns.std() * np.sqrt(12)

    sharpe = np.nan
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(12)

    drawdown = calculate_drawdown(returns)
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


strategy_metrics = calculate_metrics(all_returns["ETF Rotation Strategy"])


# ============================================================
# METRIC CARDS
# ============================================================

st.markdown('<p class="section-header">Performance Overview</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    ("CAGR", strategy_metrics["CAGR"]),
    ("Volatility", strategy_metrics["Volatility"]),
    ("Sharpe Ratio", strategy_metrics["Sharpe Ratio"]),
    ("Max Drawdown", strategy_metrics["Max Drawdown"]),
    ("Win Rate", strategy_metrics["Win Rate"])
]

for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
    if label == "Sharpe Ratio":
        display_value = f"{value:.2f}"
    else:
        display_value = f"{value:.2%}"

    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CUMULATIVE RETURNS CHART
# ============================================================

st.markdown('<p class="section-header">Strategy vs Benchmarks</p>', unsafe_allow_html=True)

fig = go.Figure()

for column in cumulative_returns.columns:
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[column],
            mode="lines",
            name=column
        )
    )

fig.update_layout(
    title="Growth of $1",
    xaxis_title="Date",
    yaxis_title="Growth Multiple",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DRAWDOWN CHART
# ============================================================

st.markdown('<p class="section-header">Drawdown Analysis</p>', unsafe_allow_html=True)

drawdown_df = pd.DataFrame()

for column in all_returns.columns:
    drawdown_df[column] = calculate_drawdown(all_returns[column])

fig_dd = go.Figure()

for column in drawdown_df.columns:
    fig_dd.add_trace(
        go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df[column],
            mode="lines",
            name=column
        )
    )

fig_dd.update_layout(
    title="Portfolio Drawdowns",
    xaxis_title="Date",
    yaxis_title="Drawdown",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig_dd, use_container_width=True)


# ============================================================
# ALLOCATION OVER TIME
# ============================================================

st.markdown('<p class="section-header">ETF Allocation Over Time</p>', unsafe_allow_html=True)

allocation_fig = px.area(
    active_weights,
    x=active_weights.index,
    y=active_weights.columns,
    title="Monthly Portfolio Allocation"
)

allocation_fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Portfolio Weight",
    height=500
)

st.plotly_chart(allocation_fig, use_container_width=True)


# ============================================================
# LATEST SIGNAL DASHBOARD
# ============================================================

st.markdown('<p class="section-header">Latest ETF Signals</p>', unsafe_allow_html=True)

latest_date = risk_adjusted_score.dropna(how="all").index[-1]

signal_dashboard = pd.DataFrame({
    "3M Momentum": mom_3m.loc[latest_date],
    "6M Momentum": mom_6m.loc[latest_date],
    "12M Momentum": mom_12m.loc[latest_date],
    "Risk-Adjusted Score": risk_adjusted_score.loc[latest_date],
    "Above 200D Moving Average": trend_signal.loc[latest_date],
    "Target Weight": target_weights.loc[latest_date]
})

signal_dashboard = signal_dashboard.sort_values("Risk-Adjusted Score", ascending=False)

st.dataframe(
    signal_dashboard.style.format({
        "3M Momentum": "{:.2%}",
        "6M Momentum": "{:.2%}",
        "12M Momentum": "{:.2%}",
        "Risk-Adjusted Score": "{:.2f}",
        "Target Weight": "{:.2%}"
    }),
    use_container_width=True
)


# ============================================================
# LATEST SCORE BAR CHART
# ============================================================

fig_scores = px.bar(
    signal_dashboard.reset_index(),
    x="Risk-Adjusted Score",
    y="index",
    orientation="h",
    title="Latest Risk-Adjusted Momentum Scores"
)

fig_scores.update_layout(
    template="plotly_dark",
    yaxis_title="ETF",
    xaxis_title="Score",
    height=500
)

st.plotly_chart(fig_scores, use_container_width=True)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.markdown('<p class="section-header">ETF Correlation Heatmap</p>', unsafe_allow_html=True)

corr = monthly_returns_aligned.corr()

fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    title="Monthly Return Correlations",
    aspect="auto"
)

fig_corr.update_layout(
    template="plotly_dark",
    height=550
)

st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# MONTHLY RETURNS HEATMAP
# ============================================================

st.markdown('<p class="section-header">Monthly Strategy Returns</p>', unsafe_allow_html=True)

heatmap_data = pd.DataFrame({
    "Year": strategy_returns.index.year,
    "Month": strategy_returns.index.month,
    "Return": strategy_returns.values
})

monthly_heatmap = heatmap_data.pivot(index="Year", columns="Month", values="Return")

fig_heatmap = px.imshow(
    monthly_heatmap,
    text_auto=".1%",
    title="Monthly Returns Heatmap",
    aspect="auto"
)

fig_heatmap.update_layout(
    template="plotly_dark",
    height=550
)

st.plotly_chart(fig_heatmap, use_container_width=True)


# ============================================================
# PROJECT EXPLANATION
# ============================================================

st.markdown('<p class="section-header">Project Explanation</p>', unsafe_allow_html=True)

st.write(
    """
    This dashboard presents a tactical ETF rotation strategy that allocates across equities,
    bonds, gold, emerging markets, and cash. The model uses momentum, trend, and volatility
    signals to rank ETFs and select the strongest risk-adjusted assets each month.

    The strategy is compared against SPY buy-and-hold and a 60/40 portfolio. The purpose is
    to study tactical asset allocation, portfolio risk, drawdowns, diversification, and
    risk-adjusted performance.
    """
)

st.info(
    "This project is for educational and research purposes only. It is not financial advice."
)


# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.markdown(
    """
    <div style="text-align:center; color:#A9B4C2;">
        Built with Python, Streamlit, Yahoo Finance, and Plotly<br>
        Created by <b>Zaeem Al Salami</b>
    </div>
    """,
    unsafe_allow_html=True
)
