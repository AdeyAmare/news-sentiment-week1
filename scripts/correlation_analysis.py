import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob


class NewsStockCorrelation:
    """
    Analyze correlation between news sentiment and stock price movements.

    This class:
        - Loads stock price files and computes daily returns
        - Loads news headlines and performs sentiment analysis
        - Aligns both datasets by date and stock symbol
        - Computes per-ticker Pearson correlations
        - Provides visualization utilities
    """

    def __init__(self, news_data_path: str, stock_data_folder: str):
        """
        Initialize paths for news and stock data.

        Args:
            news_data_path (str): Path to news CSV file.
            stock_data_folder (str): Directory containing stock CSV files.
        """
        self.news_data_path = news_data_path
        self.stock_data_folder = stock_data_folder
        self.news_df = None
        self.stock_df = None
        self.merged_df = None
        print("NewsStockCorrelation initialized.")

    # -------------------------------------------------------------------------
    def load_and_process_stocks(self):
        """
        Load and preprocess stock data.

        Steps:
            1. Load all CSV files inside the stock folder.
            2. Normalize date formats.
            3. Compute daily returns from closing prices.
            4. Combine into a single DataFrame.

        Returns:
            pd.DataFrame: Combined stock data with columns 
                ['Date', 'stock_symbol', 'daily_return'].
        """
        print("\n[1/4] Loading and processing stock data...")
        all_stocks = []

        csv_files = glob.glob(os.path.join(self.stock_data_folder, "*.csv"))
        print(f" → Found {len(csv_files)} stock files.")

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.stock_data_folder}")

        for file in csv_files:
            ticker = os.path.basename(file).replace(".csv", "")
            print(f"   - Processing {ticker}...")

            try:
                df = pd.read_csv(file)

                # Normalize dates
                df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
                df = df.sort_values("Date")

                # Daily return
                df["daily_return"] = df["Close"].pct_change()

                df["stock_symbol"] = ticker
                df_clean = df[["Date", "stock_symbol", "daily_return"]].copy()
                all_stocks.append(df_clean)

            except Exception as e:
                print(f"     ERROR while processing {ticker}: {e}")

        if not all_stocks:
            raise ValueError("Failed to load any stock data.")

        self.stock_df = pd.concat(all_stocks, ignore_index=True)
        print(f" ✔ Stock data loaded successfully ({self.stock_df.shape[0]} rows).")

        return self.stock_df

    # -------------------------------------------------------------------------
    def load_and_process_news(self):
        """
        Load and preprocess news data.

        Steps:
            1. Load news dataset.
            2. Normalize date format.
            3. Compute sentiment polarity using TextBlob.
            4. Aggregate average daily sentiment per stock.

        Returns:
            pd.DataFrame: Aggregated news sentiment with columns
                ['Date', 'stock_symbol', 'sentiment_score'].
        """
        print("\n[2/4] Loading and processing news data... (may take time)")

        try:
            df = pd.read_csv(self.news_data_path)
            print(f" → Loaded {len(df)} news articles.")

            # Date normalization
            df["Date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
            df = df.rename(columns={"stock": "stock_symbol"})

            df = df.dropna(subset=["Date", "headline"])
            print(f" → After cleaning: {len(df)} valid rows.")

            # Sentiment analysis
            print(" → Performing sentiment analysis on headlines...")
            df["sentiment_score"] = df["headline"].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )

            self.news_df = (
                df.groupby(["Date", "stock_symbol"])["sentiment_score"]
                .mean()
                .reset_index()
            )

            print(f" ✔ News data processed ({self.news_df.shape[0]} daily entries).")
            return self.news_df

        except Exception as e:
            print(f"ERROR loading news data: {e}")
            return None

    # -------------------------------------------------------------------------
    def merge_data(self):
        """
        Merge processed stock and news datasets.

        Returns:
            pd.DataFrame: Merged dataset containing
                ['Date', 'stock_symbol', 'sentiment_score', 'daily_return'].
        """
        print("\n[3/4] Merging datasets...")

        if self.stock_df is None or self.news_df is None:
            raise ValueError("Stock and news data must be loaded first.")

        self.merged_df = pd.merge(
            self.news_df, self.stock_df, on=["Date", "stock_symbol"], how="inner"
        )

        print(f" ✔ Merge completed: {self.merged_df.shape[0]} rows matched.")
        return self.merged_df

    # -------------------------------------------------------------------------
    def calculate_correlation(self):
        """
        Compute Pearson correlation between sentiment and stock returns per ticker.

        Returns:
            pd.DataFrame: Each row contains:
                - stock_symbol (str)
                - correlation (float)
                - p_value (float)
                - count (int): number of valid data points
        """
        print("\n[4/4] Calculating correlation per stock symbol...")

        if self.merged_df is None:
            self.merge_data()

        results = []

        for ticker, group in self.merged_df.groupby("stock_symbol"):
            clean = group.dropna(subset=["sentiment_score", "daily_return"])

            print(f"   - {ticker}: {len(clean)} valid rows")
            if len(clean) > 5:
                corr, p = pearsonr(clean["sentiment_score"], clean["daily_return"])
                results.append(
                    {
                        "stock_symbol": ticker,
                        "correlation": corr,
                        "p_value": p,
                        "count": len(clean),
                    }
                )

        print(" ✔ Correlation analysis completed.")
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    def plot_correlation_bar(self, correlation_df):
        """
        Plot bar chart of correlation coefficients with unique colors per ticker.
        """
        print("\nPlotting correlation bar chart...")

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=correlation_df,
            x="stock_symbol",
            y="correlation",
            hue="stock_symbol",  # ← assigns a unique color to each ticker
            dodge=False,
            palette="tab20",  # ← color set with many distinct colors
        )

        plt.title("Pearson Correlation: Daily Sentiment vs Stock Return")
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("Correlation Coefficient")
        plt.xlabel("Stock Symbol")
        plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    def plot_scatter(self, ticker: str = None):
        """
        Plot sentiment vs daily return scatter plot, color-coded by ticker.
        """
        print(f"\nPlotting scatter plot ({ticker if ticker else 'all stocks'})...")

        data = self.merged_df

        if ticker:
            data = data[data["stock_symbol"] == ticker]
            title = f"Sentiment vs Returns: {ticker}"
        else:
            title = "Sentiment vs Returns: All Stocks"

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=data,
            x="sentiment_score",
            y="daily_return",
            hue="stock_symbol",  # ← different colors per ticker
            palette="tab20",
            alpha=0.7,
        )

        plt.axhline(0, color="grey", linestyle="--")
        plt.axvline(0, color="grey", linestyle="--")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Daily Return")
        plt.title(title)
        plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    def plot_correlation_heatmap(self):
        """
        Plot a heatmap of correlations between tickers using sentiment vs return correlation.
        Rows/columns represent tickers; values are correlation coefficients.
        """
        print("\nPlotting correlation heatmap...")

        if self.merged_df is None:
            self.merge_data()

        # pivot: ticker vs ticker (sentiment-return correlation for each ticker)
        corr_df = (
            self.merged_df.groupby("stock_symbol")
            .apply(
                lambda g: pearsonr(
                    g["sentiment_score"].dropna(), g["daily_return"].dropna()
                )[0]
                if len(g.dropna()) > 5
                else np.nan
            )
            .to_frame("correlation")
        )

        # Convert to square matrix for heatmap
        heatmap_df = corr_df.pivot_table(
            values="correlation", index="stock_symbol", columns="stock_symbol"
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_df, annot=True, cmap="coolwarm", linewidths=0.5, center=0
        )

        plt.title("Correlation Heatmap: Sentiment vs Returns (Per Ticker)")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    def plot_pairplot_all(self):
        """
        Generate a pairplot (sentiment vs returns) for all tickers.
        """
        print("\nPlotting pairplot for all tickers...")

        if self.merged_df is None:
            self.merge_data()

        data = self.merged_df.dropna(subset=["sentiment_score", "daily_return"])

        sns.pairplot(
            data,
            vars=["sentiment_score", "daily_return"],
            hue="stock_symbol",
            corner=True,
            diag_kind="kde",
            palette="tab20",
        )

        plt.suptitle("Pairplot: Sentiment vs Daily Returns (All Tickers)", y=1.02)
        plt.show()

    # -------------------------------------------------------------------------
    def plot_sentiment_distribution(self, ticker=None):
        """
        Plot the distribution of sentiment scores.
        """
        data = self.news_df
        title = "All Stocks Sentiment Distribution"

        if ticker is not None:
            data = data[data["stock_symbol"] == ticker]
            title = f"{ticker} Sentiment Distribution"

        plt.figure(figsize=(10, 6))
        sns.histplot(data["sentiment_score"], bins=30, kde=True, color="skyblue")
        plt.title(title)
        plt.xlabel("Sentiment Score (-1 to 1)")
        plt.ylabel("Number of Headlines")
        plt.show()

    # -------------------------------------------------------------------------
    def plot_daily_sentiment(self, ticker=None):
        """
        Plot time series of daily average sentiment scores.
        """
        data = self.news_df
        title = "Daily Average Sentiment: All Stocks"

        if ticker is not None:
            data = data[data["stock_symbol"] == ticker]
            title = f"Daily Average Sentiment: {ticker}"

        data = data.sort_values("Date")

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="Date", y="sentiment_score", marker="o")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Average Sentiment Score")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
