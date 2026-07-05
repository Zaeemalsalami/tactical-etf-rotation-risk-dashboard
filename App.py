# ============================================================
# Tactical ETF Rotation Strategy Dashboard
# Created by Zaeem Al Salami
# CEO-Style Streamlit Dashboard
# ============================================================

import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ETF Rotation Dashboard | Zaeem Al Salami",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CEO-STYLE DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
        color: white;
    }

    .main-title {
        font-size: 44px;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #cbd5e1;
        margin-top: 4px;
    }

    .creator {
        font-size: 16px;
        color: #38bdf8;
        font-weight: 700;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .executive-card {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 12px;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 900;
        margin-top: 6px;
    }

    .positive {
        color: #22c55e;
        font-weight: 800;
    }

    .negative {
        color: #ef4444;
        font-weight: 800;
    }

    .neutral {
        color: #38bdf8;
        font-weight: 800;
    }

    div[data-testid="stMetricValue"] {
        color: white;
    }

    div[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_resample_month_end(data: pd.DataFrame, method: str = "last") -> pd.DataFrame:
    """
    Resamples data to month-end.
    Uses 'ME' for newer pandas versions and falls back to 'M' if needed.
    """
    if data.empty:
        return data

    try:
        if method == "last":
            return data.resample("ME").last()
        if method == "mean":
            return data.resample("ME").mean()
        if method == "sum":
            return data.resample("ME").sum()
    except Exception:
        if method == "last":
            return data.resample("M").last()
        if method == "mean":
            return data.resample("M").mean()
        if method == "sum":
            return data.resample("M").sum()

    raise ValueError("Invalid resampling method.")


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculates drawdown from a return series.
    """
    wealth = (1 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def calculate_metrics(returns: pd.Series) -> dict:
    """
    Calculates key investment performance metrics.
    Monthly returns are assumed.
    """
    returns = returns.dropna()

    if returns.empty:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Max Drawdown": np.nan,
            "Win Rate": np.nan,
            "Final Growth": np.nan
        }

    months = len(returns)
    cumulative_growth = (1 + returns).prod()

    cagr = cumulative_growth ** (12 / months) - 1
    volatility = returns.std() * np.sqrt(12)

    sharpe = np.nan
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(12)

    downside = returns[returns < 0]
    sortino = np.nan
    if len(downside) > 1 and downside.std() != 0:
        sortino = returns.mean() / downside.std() * np.sqrt(12)

    drawdown = calculate_drawdown(returns)
    max_drawdown = drawdown.min()
    win_rate = (returns > 0).mean()

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Final Growth": cumulative_growth
    }


def format_pct(value):
    """
    Formats decimal values as percentages.
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_num(value):
    """
    Formats numeric values.
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def executive_regime(latest_weights: pd.Series) -> str:
    """
    Creates a simple investment regime label based on latest allocation.
    """
    defensive_assets = ["BIL", "AGG", "TLT", "GLD"]
    growth_assets = ["SPY", "QQQ", "EEM"]

    defensive_weight = latest_weights.reindex(defensive_assets).fillna(0).sum()
    growth_weight = latest_weights.reindex(growth_assets).fillna(0).sum()

    if defensive_weight >= 0.70:
        return "Defensive"
    if growth_weight >= 0.70:
        return "Risk-On"
    return "Balanced"


@st.cache_data(show_spinner=False)
def download_prices(ticker_tuple, start_date):
    """
    Downloads adjusted ETF price data from Yahoo Finance.
    """
    tickers = list(ticker_tuple)

    data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        threads=True
    )

    if data is None or data.empty:
        raise ValueError("No data downloaded. Please check the ticker list or start date.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.ffill().dropna(how="all")

    valid_cols = [col for col in prices.columns if prices[col].dropna().shape[0] > 250]
    prices = prices[valid_cols].ffill().dropna()

    if prices.empty:
        raise ValueError("Cleaned price data is empty.")

    return prices


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">Tactical ETF Rotation Strategy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Executive investment dashboard for tactical asset allocation, risk monitoring, and benchmark comparison</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="creator">Created by Zaeem Al Salami</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("Dashboard Settings")

default_tickers = ["SPY", "QQQ", "TLT", "AGG", "GLD", "EEM", "BIL"]

selected_tickers = st.sidebar.multiselect(
    "Select ETF universe",
    default_tickers,
    default=default_tickers
)

start_date = st.sidebar.date_input(
    "Start date",
    value=date(2010, 1, 1)
)

top_n = st.sidebar.slider(
    "Number of ETFs selected",
    min_value=1,
    max_value=5,
    value=3
)

transaction_cost_bps = st.sidebar.slider(
    "Transaction cost in basis points",
    min_value=0,
    max_value=50,
    value=5
)

initial_capital = st.sidebar.number_input(
    "Initial capital",
    min_value=1000,
    value=100000,
    step=10000
)


# ============================================================
# INPUT VALIDATION
# ============================================================

if len(selected_tickers) < 3:
    st.error("Please select at least three ETFs.")
    st.stop()

if "BIL" not in selected_tickers:
    st.warning("BIL is not selected. The dashboard will use the first ETF as the fallback cash asset.")

cash_ticker = "BIL" if "BIL" in selected_tickers else selected_tickers[0]

if top_n >= len(selected_tickers):
    top_n = max(1, len(selected_tickers) - 1)


# ============================================================
# DATA DOWNLOAD
# ============================================================

with st.spinner("Downloading market data and building dashboard..."):
    try:
        prices = download_prices(tuple(selected_tickers), start_date)
    except Exception as error:
        st.error(f"Data download failed: {error}")
        st.stop()

available_tickers = prices.columns.tolist()

if len(available_tickers) < 3:
    st.error("Not enough valid ETF data available.")
    st.stop()


# ============================================================
# RETURNS AND FEATURES
# ============================================================

daily_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

monthly_prices = safe_resample_month_end(prices, method="last")
monthly_returns = monthly_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

mom_3m = monthly_prices.pct_change(3)
mom_6m = monthly_prices.pct_change(6)
mom_12m = monthly_prices.pct_change(12)

weighted_momentum = (0.30 * mom_3m) + (0.30 * mom_6m) + (0.40 * mom_12m)

rolling_daily_vol = daily_returns.rolling(63).std() * np.sqrt(252)
monthly_volatility = safe_resample_month_end(rolling_daily_vol, method="last")

moving_average_200 = prices.rolling(200).mean()
daily_trend = prices > moving_average_200
monthly_trend = safe_resample_month_end(daily_trend.astype(float), method="last").astype(bool)

risk_adjusted_score = weighted_momentum / monthly_volatility
risk_adjusted_score = risk_adjusted_score.replace([np.inf, -np.inf], np.nan)


# ============================================================
# TARGET WEIGHTS
# ============================================================

common_index = (
    monthly_returns.index
    .intersection(risk_adjusted_score.index)
    .intersection(monthly_trend.index)
)

monthly_returns = monthly_returns.reindex(common_index)
risk_adjusted_score = risk_adjusted_score.reindex(common_index)
monthly_trend = monthly_trend.reindex(common_index)

target_weights = pd.DataFrame(
    0.0,
    index=common_index,
    columns=available_tickers
)

risk_assets = [ticker for ticker in available_tickers if ticker != cash_ticker]

for dt in common_index:
    scores_today = risk_adjusted_score.loc[dt, risk_assets].dropna()
    trend_today = monthly_trend.loc[dt, risk_assets].reindex(scores_today.index).fillna(False)

    eligible = scores_today[(scores_today > 0) & (trend_today == True)]

    if eligible.empty:
        target_weights.loc[dt, cash_ticker] = 1.0
    else:
        selected = eligible.sort_values(ascending=False).head(top_n).index
        target_weights.loc[dt, selected] = 1.0 / len(selected)


# ============================================================
# BACKTEST
# ============================================================

active_weights = target_weights.shift(1)

backtest_index = monthly_returns.index.intersection(active_weights.dropna(how="all").index)

active_weights = active_weights.reindex(backtest_index).fillna(0)
monthly_returns_bt = monthly_returns.reindex(backtest_index)

gross_strategy_returns = (active_weights * monthly_returns_bt).sum(axis=1)

turnover = target_weights.diff().abs().sum(axis=1).shift(1)
turnover = turnover.reindex(backtest_index).fillna(0)

transaction_cost = turnover * (transaction_cost_bps / 10000)
strategy_returns = gross_strategy_returns - transaction_cost

strategy_value = initial_capital * (1 + strategy_returns).cumprod()


# ============================================================
# BENCHMARKS
# ============================================================

benchmark_returns = pd.DataFrame(index=strategy_returns.index)

if "SPY" in monthly_returns.columns:
    benchmark_returns["SPY Buy & Hold"] = monthly_returns["SPY"].reindex(strategy_returns.index)

if "SPY" in monthly_returns.columns and "AGG" in monthly_returns.columns:
    benchmark_returns["60/40 Portfolio"] = (
        0.60 * monthly_returns["SPY"].reindex(strategy_returns.index)
        + 0.40 * monthly_returns["AGG"].reindex(strategy_returns.index)
    )

all_returns = pd.concat(
    [strategy_returns.rename("ETF Rotation Strategy"), benchmark_returns],
    axis=1
).dropna()

if all_returns.empty:
    st.error("Backtest results are empty. Try an earlier start date or select more ETFs.")
    st.stop()

cumulative_returns = (1 + all_returns).cumprod()


# ============================================================
# METRICS
# ============================================================

metrics = {
    column: calculate_metrics(all_returns[column])
    for column in all_returns.columns
}

metrics_df = pd.DataFrame(metrics).T

latest_date = target_weights.index[-1]
latest_weights = target_weights.loc[latest_date]
latest_selected = latest_weights[latest_weights > 0].sort_values(ascending=False)

regime = executive_regime(latest_weights)

strategy_metrics = metrics_df.loc["ETF Rotation Strategy"]


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

summary_col1, summary_col2, summary_col3 = st.columns([1.2, 1.2, 1.6])

with summary_col1:
    st.markdown(
        f"""
        <div class="executive-card">
            <div class="metric-label">Current Market Stance</div>
            <div class="metric-value neutral">{regime}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_col2:
    st.markdown(
        f"""
        <div class="executive-card">
            <div class="metric-label">Latest Signal Date</div>
            <div class="metric-value">{latest_date.date()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_col3:
    allocation_text = "<br>".join([f"{ticker}: {weight:.1%}" for ticker, weight in latest_selected.items()])
    st.markdown(
        f"""
        <div class="executive-card">
            <div class="metric-label">Current Model Allocation</div>
            <div style="font-size:22px; font-weight:800; color:white; margin-top:8px;">
                {allocation_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# KPI CARDS
# ============================================================

st.markdown('<div class="section-title">Strategy Performance KPIs</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi_values = [
    ("CAGR", format_pct(strategy_metrics["CAGR"])),
    ("Volatility", format_pct(strategy_metrics["Volatility"])),
    ("Sharpe Ratio", format_num(strategy_metrics["Sharpe"])),
    ("Max Drawdown", format_pct(strategy_metrics["Max Drawdown"])),
    ("Win Rate", format_pct(strategy_metrics["Win Rate"]))
]

for col, (label, value) in zip([kpi1, kpi2, kpi3, kpi4, kpi5], kpi_values):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PERFORMANCE CHART
# ============================================================

st.markdown('<div class="section-title">Growth of $1: Strategy vs Benchmarks</div>', unsafe_allow_html=True)

fig_growth = go.Figure()

for column in cumulative_returns.columns:
    fig_growth.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[column],
            mode="lines",
            name=column,
            line=dict(width=3)
        )
    )

fig_growth.update_layout(
    template="plotly_dark",
    height=560,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    title="Cumulative Return Comparison",
    xaxis_title="Date",
    yaxis_title="Growth of $1",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_growth, use_container_width=True)


# ============================================================
# DRAWDOWN CONTROL
# ============================================================

st.markdown('<div class="section-title">Drawdown Control</div>', unsafe_allow_html=True)

drawdown_df = pd.DataFrame({
    column: calculate_drawdown(all_returns[column])
    for column in all_returns.columns
})

fig_dd = go.Figure()

for column in drawdown_df.columns:
    fig_dd.add_trace(
        go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df[column],
            mode="lines",
            name=column,
            line=dict(width=3)
        )
    )

fig_dd.update_layout(
    template="plotly_dark",
    height=600,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    title="Drawdown Comparison",
    xaxis_title="Date",
    yaxis_title="Drawdown",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_dd, use_container_width=True)


# ============================================================
# ROLLING VOLATILITY
# ============================================================

st.markdown('<div class="section-title">Rolling Volatility</div>', unsafe_allow_html=True)

rolling_vol = all_returns.rolling(12).std() * np.sqrt(12)

fig_vol = go.Figure()

for column in rolling_vol.columns:
    fig_vol.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[column],
            mode="lines",
            name=column,
            line=dict(width=3)
        )
    )

fig_vol.update_layout(
    template="plotly_dark",
    height=600,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    title="Rolling 12-Month Annualized Volatility",
    xaxis_title="Date",
    yaxis_title="Annualized Volatility",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_vol, use_container_width=True)


# ============================================================
# ALLOCATION AND LATEST SIGNALS
# ============================================================

st.markdown('<div class="section-title">ETF Allocation Over Time</div>', unsafe_allow_html=True)

fig_alloc = px.area(
    active_weights,
    x=active_weights.index,
    y=active_weights.columns,
    title="Monthly Tactical Allocation"
)

fig_alloc.update_layout(
    template="plotly_dark",
    height=520,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    xaxis_title="Date",
    yaxis_title="Portfolio Weight",
    hovermode="x unified",
    legend=dict(orientation="h")
)

st.plotly_chart(fig_alloc, use_container_width=True)


# ============================================================
# SIGNAL DASHBOARD
# ============================================================

st.markdown('<div class="section-title">Latest ETF Signal Dashboard</div>', unsafe_allow_html=True)

signal_date = risk_adjusted_score.dropna(how="all").index[-1]

signal_dashboard = pd.DataFrame({
    "3M Momentum": mom_3m.loc[signal_date],
    "6M Momentum": mom_6m.loc[signal_date],
    "12M Momentum": mom_12m.loc[signal_date],
    "Annualized Volatility": monthly_volatility.loc[signal_date],
    "Risk-Adjusted Score": risk_adjusted_score.loc[signal_date],
    "Above 200D MA": monthly_trend.loc[signal_date],
    "Target Weight": target_weights.loc[signal_date]
})

signal_dashboard = signal_dashboard.sort_values("Risk-Adjusted Score", ascending=False)

st.dataframe(
    signal_dashboard.style.format({
        "3M Momentum": "{:.2%}",
        "6M Momentum": "{:.2%}",
        "12M Momentum": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Risk-Adjusted Score": "{:.2f}",
        "Target Weight": "{:.2%}"
    }),
    use_container_width=True
)


fig_score = px.bar(
    signal_dashboard.reset_index(),
    x="Risk-Adjusted Score",
    y="index",
    orientation="h",
    title="Latest Risk-Adjusted Momentum Ranking",
    text="Risk-Adjusted Score"
)

fig_score.update_traces(texttemplate="%{text:.2f}", textposition="outside")

fig_score.update_layout(
    template="plotly_dark",
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    xaxis_title="Risk-Adjusted Score",
    yaxis_title="ETF"
)

st.plotly_chart(fig_score, use_container_width=True)


# ============================================================
# CORRELATION AND MONTHLY RETURNS
# ============================================================

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown('<div class="section-title">ETF Correlation Matrix</div>', unsafe_allow_html=True)

    corr = monthly_returns_bt.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Monthly Return Correlations"
    )

    fig_corr.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.85)"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

with chart_col4:
    st.markdown('<div class="section-title">Monthly Return Heatmap</div>', unsafe_allow_html=True)

    heatmap_data = pd.DataFrame({
        "Year": strategy_returns.index.year,
        "Month": strategy_returns.index.month,
        "Return": strategy_returns.values
    })

    monthly_heatmap = heatmap_data.pivot(index="Year", columns="Month", values="Return")

    fig_heatmap = px.imshow(
        monthly_heatmap,
        text_auto=".1%",
        aspect="auto",
        title="Strategy Monthly Returns"
    )

    fig_heatmap.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.85)"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)


# ============================================================
# PERFORMANCE TABLE
# ============================================================

st.markdown('<div class="section-title">Full Performance Summary</div>', unsafe_allow_html=True)

display_metrics = metrics_df.copy()

percent_cols = ["CAGR", "Volatility", "Max Drawdown", "Win Rate"]
number_cols = ["Sharpe", "Sortino", "Final Growth"]

for col in percent_cols:
    display_metrics[col] = display_metrics[col].map(format_pct)

for col in number_cols:
    display_metrics[col] = display_metrics[col].map(format_num)

st.dataframe(display_metrics, use_container_width=True)


# ============================================================
# CEO INTERPRETATION
# ============================================================

st.markdown('<div class="section-title">Executive Interpretation</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="executive-card">
    <b>Current positioning:</b> The model is currently classified as <span class="neutral">{regime}</span> based on its latest ETF allocation.<br><br>

    <b>Strategy logic:</b> The dashboard ranks ETFs using momentum, trend, and volatility. Assets must show positive risk-adjusted momentum and trade above their 200-day moving average to qualify for allocation.<br><br>

    <b>Risk discipline:</b> The model avoids lookahead bias by using month-end signals for the following month's allocation. It also includes transaction cost assumptions when portfolio weights change.<br><br>

    <b>Business value:</b> This dashboard helps decision-makers quickly understand whether the model favors growth assets, defensive assets, or cash, while comparing performance against SPY and a traditional 60/40 portfolio.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.write("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; font-size:14px;">
        Built with Python, Streamlit, Yahoo Finance, Pandas, NumPy, and Plotly<br>
        Created by <b style="color:#38bdf8;">Zaeem Al Salami</b><br><br>
        Educational investment research project. Not financial advice.
    </div>
    """,
    unsafe_allow_html=True
)
