import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from typing import Dict, Optional


class TechnicalAnalyzer:
    """
    Compute technical indicators and key financial metrics from OHLCV data,
    including PyNance-style metrics.
    
    Expected DataFrame columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize TechnicalAnalyzer with OHLCV data.

        Args:
            df (pd.DataFrame): Stock OHLCV data with a 'Date' column.
        """
        self.df = df.copy()

        # Ensure 'Date' is datetime and set as index
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df.set_index("Date", inplace=True)

        # Ensure numeric columns
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        self.df[numeric_cols] = self.df[numeric_cols].astype(float)

    def apply_talib_indicators(self) -> pd.DataFrame:
        """
        Compute common technical indicators using TA-Lib:
            - SMA 20 and 50
            - RSI 14
            - MACD, MACD Signal, MACD Histogram

        Returns:
            pd.DataFrame: DataFrame with new indicator columns added.
        """
        print("[INFO] Calculating TA-Lib technical indicators...")

        self.df["SMA_20"] = talib.SMA(self.df["Close"], timeperiod=20)
        self.df["SMA_50"] = talib.SMA(self.df["Close"], timeperiod=50)
        self.df["RSI"] = talib.RSI(self.df["Close"], timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(
            self.df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.df["MACD"] = macd
        self.df["MACD_Signal"] = macd_signal
        self.df["MACD_Hist"] = macd_hist

        print("[INFO] Indicators added: SMA_20, SMA_50, RSI, MACD components.")
        return self.df

    def calculate_financial_metrics(self, risk_free_rate: float = 0.01) -> Dict[str, float]:
        """
        Compute key financial metrics and PyNance-style metrics:
            - Annualized volatility
            - Sharpe ratio
            - Maximum drawdown
            - Cumulative returns
            - Total return
            - Rolling volatility

        Args:
            risk_free_rate (float): Annual risk-free rate (default 1%).

        Returns:
            Dict[str, float]: Financial and risk metrics.
        """
        if self.df is None or self.df.empty:
            raise ValueError("TechnicalAnalyzer: DataFrame is empty.")

        metrics: Dict[str, float] = {}

        # Daily returns
        self.df["Daily_Return"] = self.df["Close"].pct_change()

        # Cumulative returns (PyNance style)
        self.df["Cumulative_Return"] = (1 + self.df["Daily_Return"]).cumprod()

        # Annualized volatility
        metrics["volatility"] = self.df["Daily_Return"].std() * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = self.df["Daily_Return"] - (risk_free_rate / 252)
        metrics["sharpe_ratio"] = (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        )

        # Maximum drawdown
        cumulative = self.df["Cumulative_Return"]
        peak = cumulative.expanding().max()
        drawdowns = cumulative / peak - 1
        metrics["max_drawdown"] = drawdowns.min()

        # Total cumulative return
        metrics["total_return"] = self.df["Cumulative_Return"].iloc[-1] - 1

        # Rolling volatility (20-day annualized)
        self.df["Rolling_Vol_20"] = self.df["Daily_Return"].rolling(20).std() * np.sqrt(252)

        return metrics


class StockVisualizer:
    """
    Visualize stock OHLCV data, technical indicators, and PyNance-style metrics.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize StockVisualizer.

        Args:
            df (pd.DataFrame): Stock OHLCV data with indicators and metrics.
        """
        self.df = df.copy()

    def plot_price_and_ma(self):
        """Plot Close price along with SMA 20 and SMA 50."""
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
        """Plot RSI with overbought (70) and oversold (30) reference lines."""
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
        """Plot MACD line, signal line, and histogram."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["MACD"], label="MACD", color="blue")
        plt.plot(self.df.index, self.df["MACD_Signal"], label="Signal", color="red")
        plt.bar(
            self.df.index,
            self.df["MACD_Hist"],
            label="Histogram",
            color="gray",
            alpha=0.3
        )
        plt.title("MACD (Moving Average Convergence Divergence)")
        plt.xlabel("Date")
        plt.ylabel("MACD Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_cumulative_returns(self):
        """Plot cumulative returns over time (PyNance style)."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["Cumulative_Return"], label="Cumulative Return", color="blue")
        plt.title("Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_rolling_volatility(self):
        """Plot rolling 20-day annualized volatility (PyNance style)."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["Rolling_Vol_20"], label="Rolling Volatility (20 days)", color="orange")
        plt.title("Rolling Annualized Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
