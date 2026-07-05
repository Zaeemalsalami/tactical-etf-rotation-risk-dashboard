# Tactical ETF Rotation Strategy with Risk Dashboard

## Overview

This project builds a Python-based tactical ETF rotation strategy that allocates across equities, bonds, gold, emerging markets, and cash using momentum, trend, and volatility signals.

The main goal of the strategy is to generate competitive returns while controlling downside risk and minimizing major portfolio drawdowns.

The strategy is backtested against SPY buy-and-hold and a traditional 60/40 portfolio. The project also includes a Streamlit dashboard designed to help users understand portfolio allocation, risk, drawdowns, volatility, and benchmark performance.

## Live Dashboard

View the interactive dashboard here:

https://etf-rotation-dashboard-zaeem.streamlit.app

## Created By

Zaeem Al Salami

## Strategy Objective

The objective of this tactical ETF rotation strategy is not only to maximize return, but to improve risk-adjusted performance.

The model aims to capture upside opportunities across asset classes while reducing downside exposure by rotating into defensive assets or cash when market conditions weaken.

In simple terms, the strategy tries to answer:

Should the portfolio currently favor growth assets, defensive assets, or cash?

## ETF Universe

The model uses the following ETFs:

- SPY: U.S. equities
- QQQ: Technology stocks
- TLT: Long-term Treasury bonds
- AGG: Broad U.S. bond market
- GLD: Gold
- EEM: Emerging markets
- BIL: Cash / Treasury bills

## Strategy Logic

The model follows a systematic tactical allocation process:

1. Downloads ETF price data from Yahoo Finance.
2. Calculates monthly returns.
3. Calculates 3-month, 6-month, and 12-month momentum.
4. Builds a weighted momentum score.
5. Applies a 200-day moving average trend filter.
6. Calculates rolling volatility.
7. Ranks ETFs using risk-adjusted momentum.
8. Selects the strongest ETFs each month.
9. Moves to cash when no asset passes the filter.
10. Shifts portfolio weights forward to avoid lookahead bias.
11. Backtests performance with transaction cost assumptions.
12. Compares results against SPY and a 60/40 benchmark.

## Key Features

- Tactical ETF rotation model
- Monthly rebalancing
- Momentum and trend-following signals
- Volatility-adjusted ranking
- Defensive allocation to cash when risk conditions weaken
- Transaction cost assumptions
- No-lookahead backtesting
- Strategy vs SPY comparison
- Strategy vs 60/40 portfolio comparison
- Drawdown analysis
- Rolling volatility analysis
- ETF allocation dashboard
- Correlation analysis
- Monthly return heatmap
- Streamlit web dashboard

## Performance Metrics

The project calculates:

- CAGR
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Final growth multiple
- Turnover
- Risk-adjusted performance

## Dashboard Purpose

The dashboard helps investors and analysts understand whether the model currently favors growth assets, defensive assets, or cash.

Investors can use the dashboard as a decision-support tool to study tactical asset allocation, portfolio risk, downside protection, and historical strategy behavior.

This project is not intended to be automatic trading advice. It is an educational investment research project.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Yahoo Finance / yfinance
- Plotly
- GitHub
- Google Colab

## Files in This Repository

- `App.py`: Streamlit dashboard application
- `Untitled1.ipynb`: Original Google Colab notebook
- `requirements.txt`: Required Python libraries
- `README.md`: Project documentation
- `charts/`: Saved charts and visualizations
- `data/`: Strategy data outputs

## Resume Bullet

Built a Python-based tactical ETF rotation model designed to generate competitive returns while controlling downside risk and minimizing drawdowns. The strategy allocates across equities, bonds, gold, emerging markets, and cash using momentum, trend, and volatility signals; backtested performance against SPY and a 60/40 portfolio, incorporated transaction costs, avoided lookahead bias, and developed an interactive Streamlit risk dashboard with CAGR, Sharpe ratio, volatility, drawdown, allocation, and correlation analytics.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice.
