import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import pynance as pn  # Ensure pynance is installed: pip install pynance
import numpy as np

class TechnicalAnalyzer:
    def __init__(self, df):
        """
        Expects a DataFrame with columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
        Ensures 'Date' is the index.
        """
        self.df = df.copy()
        
        # Ensure Date is datetime and set as index if not already
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
        
        # Ensure columns are floats
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.df[cols] = self.df[cols].astype(float)

    def apply_talib_indicators(self):
        """
        Applies TA-Lib indicators: SMA, RSI, MACD.
        """
        print("\n[1] Calculating Technical Indicators (TA-Lib)...")
        
        # 1. Simple Moving Average (SMA)
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['SMA_50'] = talib.SMA(self.df['Close'], timeperiod=50)
        
        # 2. Relative Strength Index (RSI)
        self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=14)
        
        # 3. MACD (Moving Average Convergence Divergence)
        # macd = fast_ema - slow_ema
        # macdsignal = ema(macd)
        # macdhist = macd - macdsignal
        macd, macdsignal, macdhist = talib.MACD(
            self.df['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macdsignal
        self.df['MACD_Hist'] = macdhist
        
        print("Indicators added: SMA_20, SMA_50, RSI, MACD variables.")
        return self.df

    def calculate_financial_metrics(self):
        """Calculate key financial metrics using PyNance

        Returns:
            dict: Dictionary containing calculated financial metrics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        metrics = {}

        # Calculate daily returns
        self.df['Daily_Return'] = self.df['Close'].pct_change()

        # Calculate volatility (standard deviation of returns)
        metrics['volatility'] = self.df['Daily_Return'].std() * np.sqrt(252)  # Annualized

        # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
        risk_free_rate = 0.01
        excess_returns = self.df['Daily_Return'] - risk_free_rate/252
        metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Calculate maximum drawdown
        cum_returns = (1 + self.df['Daily_Return']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns/rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()

        return metrics




class StockVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_price_and_ma(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.5)
        plt.plot(self.df.index, self.df['SMA_20'], label='SMA 20', color='orange')
        plt.plot(self.df.index, self.df['SMA_50'], label='SMA 50', color='green')
        plt.title('Stock Price vs Moving Averages')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_rsi(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', alpha=0.5, color='red')
        plt.axhline(30, linestyle='--', alpha=0.5, color='green')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_macd(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df.index, self.df['MACD_Signal'], label='Signal Line', color='red')
        plt.bar(self.df.index, self.df['MACD_Hist'], label='Histogram', color='gray', alpha=0.3)
        plt.title('MACD (Moving Average Convergence Divergence)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()