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