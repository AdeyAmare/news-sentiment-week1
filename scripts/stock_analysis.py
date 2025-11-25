import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from pynance import portfolio_optimizer as po  # NEW Integration
from typing import Dict, Optional


class TechnicalAnalyzer:
    """
    Compute technical indicators and key financial metrics from OHLCV data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize TechnicalAnalyzer with OHLCV data.
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
        Compute common technical indicators using TA-Lib.
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

    def calculate_financial_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Compute key financial metrics + PyNance portfolio optimizer outputs.
        """
        if self.df is None or self.df.empty:
            raise ValueError("TechnicalAnalyzer: DataFrame is empty.")

        metrics: Dict[str, float] = {}

        # Daily returns
        self.df["Daily_Return"] = self.df["Close"].pct_change().fillna(0)

        # Cumulative returns
        self.df["Cumulative_Return"] = (1 + self.df["Daily_Return"]).cumprod()

        # ==========================================================
        # NEW PYNANCE PORTFOLIO OPTIMIZER LOGIC
        # ==========================================================
        try:
            print("[INFO] Running PyNance portfolio optimizer calculations...")

            TICKERS = ["AAPL", "MSFT", "META", "NVDA", "TSLA"]
            portfolio = po.PortfolioCalculations(TICKERS)

            # Max Sharpe Portfolio
            metrics["max_sharpe_rr"] = portfolio.max_sharpe_portfolio("rr")
            metrics["max_sharpe_weights"] = portfolio.max_sharpe_portfolio("df")

            # Minimum Variance Portfolio
            metrics["min_var_rr"] = portfolio.min_var_portfolio("rr")
            metrics["min_var_weights"] = portfolio.min_var_portfolio("df")

            print("[INFO] PyNance portfolio optimizer metrics added.")

        except Exception as e:
            print("[WARNING] PyNance portfolio optimizer failed:", str(e))
            print("[WARNING] Skipping PyNance portfolio metrics.")

        # ==========================================================
        # MANUAL FINANCIAL METRICS 
        # ==========================================================

        # Volatility & Sharpe
        daily = self.df["Daily_Return"]
        metrics["volatility"] = daily.std() * np.sqrt(252)

        excess = daily - (risk_free_rate / 252)
        metrics["sharpe_ratio"] = np.sqrt(252) * excess.mean() / excess.std()

        # Maximum Drawdown
        cumulative = self.df["Cumulative_Return"]
        peak = cumulative.expanding().max()
        drawdowns = cumulative / peak - 1
        metrics["max_drawdown"] = drawdowns.min()

        # Total return
        metrics["total_return"] = cumulative.iloc[-1] - 1

        # Rolling volatility (20-day)
        self.df["Rolling_Vol_20"] = daily.rolling(20).std() * np.sqrt(252)

        return metrics


class StockVisualizer:
    def __init__(self, df: pd.DataFrame, metrics: Dict):
        self.df = df.copy()
        self.metrics = metrics

    def plot_price_and_ma(self):
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
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["Cumulative_Return"], label="Cumulative Return", color="blue")
        plt.title("Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_rolling_volatility(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["Rolling_Vol_20"], label="Rolling Volatility (20 days)", color="orange")
        plt.title("Rolling Annualized Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_portfolio_weights(self, type: str = "max_sharpe"):
        """
        Plot portfolio weights returned from PyNance.

        type = "max_sharpe" or "min_var"
        """
        key = "max_sharpe_weights" if type == "max_sharpe" else "min_var_weights"

        # Extract the df returned by PyNance
        weights_df = self.metrics.get(key)
        if weights_df is None:
            print(f"[ERROR] No {type} weights found in metrics.")
            return

        # FIX: Convert DataFrame row → 1D Series
        if isinstance(weights_df, pd.DataFrame):
            # PyNance always returns ONE row, so we take row 0
            weights = weights_df.iloc[0]
        else:
            weights = weights_df  # if already Series

        plt.figure(figsize=(10, 5))
        plt.bar(weights.index, weights.values)
        plt.title(f"{type.replace('_', ' ').title()} Portfolio Weights")
        plt.ylabel("Weight")
        plt.grid(alpha=0.3)
        plt.show()
