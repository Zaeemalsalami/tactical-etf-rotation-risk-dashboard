
# Tactical ETF Rotation Strategy with Risk Dashboard

## Overview

This project builds a tactical ETF rotation strategy in Python. The model allocates across equities, bonds, gold, emerging markets, and cash using momentum, trend, and volatility signals.

The strategy is backtested against SPY and a 60/40 portfolio, includes transaction costs, avoids lookahead bias, and produces a professional risk dashboard.

## ETF Universe

- SPY: U.S. equities
- QQQ: technology equities
- TLT: long-term Treasury bonds
- AGG: aggregate bonds
- GLD: gold
- EEM: emerging markets
- BIL: cash / T-bills

## Strategy Logic

1. Download ETF prices from Yahoo Finance.
2. Calculate monthly returns.
3. Calculate 3-month, 6-month, and 12-month momentum.
4. Calculate a weighted momentum score.
5. Apply a 200-day moving average trend filter.
6. Calculate rolling annualized volatility.
7. Rank ETFs by risk-adjusted momentum.
8. Select the top ETFs monthly.
9. Allocate to cash if no ETF qualifies.
10. Shift weights by one month to avoid lookahead bias.
11. Backtest strategy returns with transaction costs.
12. Compare against SPY and 60/40 benchmarks.

## Performance Metrics

The project calculates:

- CAGR
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Calmar ratio
- Win rate
- Beta vs SPY
- Alpha vs SPY
- Monthly VaR
- Monthly CVaR
- Turnover

## Visualizations

The notebook generates:

- Cumulative return chart
- Drawdown chart
- ETF allocation over time
- Latest momentum score chart
- Rolling Sharpe ratio
- Rolling volatility
- ETF correlation heatmap
- Monthly returns heatmap
- Interactive Plotly performance chart

## Resume Bullet

Built a tactical ETF rotation model in Python that allocates across equities, bonds, gold, emerging markets, and cash using momentum, trend, and volatility signals; backtested performance against SPY and a 60/40 portfolio with transaction costs and created a risk dashboard with CAGR, Sharpe ratio, Sortino ratio, max drawdown, beta, alpha, VaR, CVaR, turnover, and correlation analytics.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice.
