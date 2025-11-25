import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from pynance import portfolio_optimizer as po
from typing import Dict, Optional


class TechnicalAnalyzer:
    """
    Computes technical indicators and key financial metrics from OHLCV data.

    Attributes:
        df (pd.DataFrame): OHLCV dataset with datetime index.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize a TechnicalAnalyzer instance.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.
        """
        self.df = df.copy()

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df.set_index("Date", inplace=True)

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        self.df[numeric_cols] = self.df[numeric_cols].astype(float)

    def apply_talib_indicators(self) -> pd.DataFrame:
        """
        Computes common TA-Lib technical indicators.

        Returns:
            pd.DataFrame: DataFrame containing newly added indicators.
        """
        print("[INFO] Calculating TA-Lib technical indicators...")

        self.df["SMA_20"] = talib.SMA(self.df["Close"], timeperiod=20)
        self.df["SMA_50"] = talib.SMA(self.df["Close"], timeperiod=50)
        self.df["RSI"] = talib.RSI(self.df["Close"], timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(
            self.df["Close"],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )

        self.df["MACD"] = macd
        self.df["MACD_Signal"] = macd_signal
        self.df["MACD_Hist"] = macd_hist

        print("[INFO] Indicators added: SMA_20, SMA_50, RSI, MACD components.")
        return self.df

    def calculate_financial_metrics(
        self, risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """
        Computes financial metrics and PyNance optimizer results.

        Args:
            risk_free_rate (float): Annual risk-free rate used for Sharpe ratio.

        Returns:
            Dict[str, float]: Dictionary containing all computed metrics.
        """
        if self.df is None or self.df.empty:
            raise ValueError("TechnicalAnalyzer: DataFrame is empty.")

        metrics: Dict[str, float] = {}

        # Daily returns
        self.df["Daily_Return"] = self.df["Close"].pct_change().fillna(0)

        # Cumulative returns
        self.df["Cumulative_Return"] = (1 + self.df["Daily_Return"]).cumprod()

        # ----------------------------------------------------------
        # PyNance Portfolio Optimizer
        # ----------------------------------------------------------
        try:
            print("[INFO] Running PyNance portfolio optimizer calculations...")

            tickers = ["AAPL", "MSFT", "META", "NVDA", "TSLA"]
            portfolio = po.PortfolioCalculations(tickers)

            metrics["max_sharpe_rr"] = portfolio.max_sharpe_portfolio("rr")
            metrics["max_sharpe_weights"] = portfolio.max_sharpe_portfolio("df")

            metrics["min_var_rr"] = portfolio.min_var_portfolio("rr")
            metrics["min_var_weights"] = portfolio.min_var_portfolio("df")

            print("[INFO] PyNance portfolio optimizer metrics added.")

        except Exception as error:
            print("[WARNING] PyNance portfolio optimizer failed:", str(error))
            print("[WARNING] Skipping PyNance portfolio metrics.")

        # ----------------------------------------------------------
        # Manual Financial Metrics
        # ----------------------------------------------------------
        daily = self.df["Daily_Return"]
        metrics["volatility"] = daily.std() * np.sqrt(252)

        excess = daily - (risk_free_rate / 252)
        metrics["sharpe_ratio"] = (
            np.sqrt(252) * excess.mean() / excess.std()
        )

        cumulative = self.df["Cumulative_Return"]
        peak = cumulative.expanding().max()
        drawdowns = cumulative / peak - 1
        metrics["max_drawdown"] = drawdowns.min()

        metrics["total_return"] = cumulative.iloc[-1] - 1

        # Rolling Volatility
        self.df["Rolling_Vol_20"] = (
            daily.rolling(20).std() * np.sqrt(252)
        )

        return metrics


class StockVisualizer:
    """
    Visualization utilities for stock analysis results.

    Attributes:
        df (pd.DataFrame): OHLCV + indicators dataset.
        metrics (dict): Dictionary containing computed financial metrics.
    """

    def __init__(self, df: pd.DataFrame, metrics: Dict):
        """
        Initialize StockVisualizer.

        Args:
            df (pd.DataFrame): DataFrame containing computed indicators.
            metrics (Dict): Financial metrics and PyNance outputs.
        """
        self.df = df.copy()
        self.metrics = metrics

    def plot_price_and_ma(self):
        """Plots closing price alongside moving averages."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df["Close"], label="Close Price", alpha=0.6)
        plt.plot(self.df.index, self.df["SMA_20"], label="SMA 20", color="orange")
        plt.plot(self.df.index, self.df["SMA_50"], label="SMA 50", color="green")
        plt.title("Close Price vs Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_rsi(self):
        """Plots the Relative Strength Index (RSI)."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["RSI"], label="RSI", color="purple")
        plt.axhline(70, color="red", linestyle="--", alpha=0.5)
        plt.axhline(30, color="green", linestyle="--", alpha=0.5)
        plt.title("Relative Strength Index (RSI)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_macd(self):
        """Plots the MACD, signal line, and histogram."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["MACD"], label="MACD", color="blue")
        plt.plot(self.df.index, self.df["MACD_Signal"], label="Signal", color="red")
        plt.bar(
            self.df.index,
            self.df["MACD_Hist"],
            label="Histogram",
            color="gray",
            alpha=0.3,
        )
        plt.title("MACD (Moving Average Convergence Divergence)")
        plt.xlabel("Date")
        plt.ylabel("MACD Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_cumulative_returns(self):
        """Plots cumulative returns."""
        plt.figure(figsize=(14, 5))
        plt.plot(
            self.df.index,
            self.df["Cumulative_Return"],
            label="Cumulative Return",
            color="blue",
        )
        plt.title("Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_rolling_volatility(self):
        """Plots rolling 20-day annualized volatility."""
        plt.figure(figsize=(14, 5))
        plt.plot(
            self.df.index,
            self.df["Rolling_Vol_20"],
            label="Rolling Volatility (20 days)",
            color="orange",
        )
        plt.title("Rolling Annualized Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_portfolio_weights(self, type: str = "max_sharpe"):
        """
        Plots PyNance-produced portfolio weights.

        Args:
            type (str): Portfolio type to plot. 
                Accepts "max_sharpe" or "min_var".
        """
        key = (
            "max_sharpe_weights"
            if type == "max_sharpe"
            else "min_var_weights"
        )

        weights_df = self.metrics.get(key)
        if weights_df is None:
            print(f"[ERROR] No {type} weights found in metrics.")
            return

        if isinstance(weights_df, pd.DataFrame):
            weights = weights_df.iloc[0]
        else:
            weights = weights_df

        plt.figure(figsize=(10, 5))
        plt.bar(weights.index, weights.values)
        plt.title(f"{type.replace('_', ' ').title()} Portfolio Weights")
        plt.ylabel("Weight")
        plt.grid(alpha=0.3)
        plt.show()
