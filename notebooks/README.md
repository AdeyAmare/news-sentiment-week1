# Notebooks Overview

### Analysis Notebooks

This directory contains Jupyter notebooks used for exploratory analysis, visualization, and technical study of the datasets.

### Available Notebooks

#### 1. ```financial_analysis_eda.ipynb```

Focus: Qualitative Analysis of Financial News

Input: ```raw_analyst_ratings.csv``` (Financial News & Stock Price Integration Dataset)

Key Operations:

Loads and cleans over 1.4 million news headlines.

Analyzes publication frequency by time of day (market hours) and day of week.

Performs Topic Modeling to categorize news into themes like "Earnings," "Upgrades," and "FDA Approvals."

Visualizes top publishers and their specific content niches.

#### 2. ```[TICKER]_stock_technical_analysis.ipynb```

Focus: Quantitative Analysis of Stock Prices

Input: ```yfinance_data/[TICKER].csv``` (or similar ticker data)

Key Operations:

Trend Analysis: Calculates Simple Moving Averages (SMA) to determine bullish/bearish trends.

Momentum Analysis: Computes RSI to identify overbought/oversold conditions.

Signal Generation: Generates MACD (Moving Average Convergence Divergence) values.

Risk Metrics: Calculates Sharpe Ratio and Volatility to assess investment risk.

Visualization: Produces professional financial charts for all indicators.

#### 3. correlation_analysis.ipynb

Focus: Correlation Analysis Between News Sentiment and Stock Performance

Input:

```processed_financial_news.csv``` (daily sentiment scores by stock)

```yfinance_data/[TICKER].csv``` (stock data)

Key Operations:

Computes daily sentiment scores from financial news headlines using TextBlob.

Aligns sentiment data with corresponding stock returns per ticker.

Calculates Pearson correlation coefficients and p-values for each stock.

Visualizes correlations with bar charts, scatter plots, heatmaps, and pairplots.

Helps identify whether news sentiment has measurable impact on stock price movements.

#### How to Run

Run ```financial_analysis_eda.ipynb``` first to understand the news context.

Run ```[TICKER]_stock_technical_analysis.ipynb``` to analyze market performance.

Run ```correlation_analysis.ipynb``` to analyze market performance.