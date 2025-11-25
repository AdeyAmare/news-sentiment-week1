# Scripts Overview

### Analysis Modules & Scripts

This directory contains the core logic and classes used to process data, calculate metrics, and generate visualizations. These modules are imported by the Jupyter notebooks.

Modules

### 1. ```data_loader.py```

#### Class: ```DataLoader```

Purpose: Handles the loading of raw data from CSV files.

Key Methods:

load_data(filepath): Reads a CSV file into a Pandas DataFrame.

### 2. ```financial_analysis.py```

#### Class: ```FinancialDataAnalyzer```

Purpose: Performs cleaning, preprocessing, and EDA on financial news datasets.

Key Methods:

```clean_data()```: Handles date parsing, deduplication, and text cleaning.

```explore_data()```: Prints descriptive statistics and missing value reports.

```analyze_publishers()```: Aggregates article counts by publisher and extracts email domains.

```extract_topics(num_topics)```: Uses LDA (Latent Dirichlet Allocation) to find hidden themes in headlines.

```analyze_key_phrases()```: Counts frequencies of specific financial terms (e.g., "FDA Approval", "Earnings").

#### Class: ```FinancialDataVisualizer```

Purpose: Generates plots for the news dataset.

Key Methods:

```descriptive_statistics()```: Plots headline lengths, publisher counts, and publication timelines.

```plot_publisher_types()```: Stacked bar charts showing the type of news different publishers focus on.

### 3. ```stock_analysis.py```

#### Class: ```TechnicalAnalyzer```

Purpose: Performs quantitative technical analysis on stock price data (OHLCV).

Key Methods:

```apply_talib_indicators()```: Calculates SMA (20/50), RSI (14), and MACD using the TA-Lib library.

```calculate_financial_metrics()```: Computes Annualized Volatility, Sharpe Ratio, and Max Drawdown using PyNance.

#### Class: ```StockVisualizer```

Purpose: Visualizes financial data and indicators.

Key Methods:

```plot_price_and_ma()```: Overlays Close Price with SMA-20 and SMA-50.

```plot_rsi()```: Plots Relative Strength Index with overbought/oversold bounds.

```plot_macd()```: Visualizes MACD line, Signal line, and Histogram.

### 4. ```correlation_analysis.py```

#### Class: ```NewsStockCorrelation```

Purpose: Analyzes the relationship between financial news sentiment and stock price movements.

Key Methods:

```load_and_process_stocks()```: Loads stock price CSVs, normalizes dates, and computes daily returns.

```load_and_process_news()```: Loads news headlines, normalizes dates, and computes average daily sentiment per stock.

```merge_data()```: Aligns stock and news data by date and ticker.

```calculate_correlation()```: Computes Pearson correlation between sentiment scores and daily stock returns per ticker.

```plot_correlation_bar()```: Visualizes correlation coefficients as a bar chart.

```plot_scatter()```: Scatter plot of sentiment vs returns.

```plot_correlation_heatmap()```: Heatmap of sentiment-return correlations across tickers.

```plot_pairplot_all()```: Pairplot for all tickers showing sentiment vs returns.

```plot_sentiment_distribution()```: Distribution of sentiment scores.

```plot_daily_sentiment()```: Time series of daily average sentiment per ticker.