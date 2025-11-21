import pandas as pd

class DataLoader:
    def __init__(self):
        self.df = None

    def load_data(self, filepath):
        """Load data from CSV file dynamically"""
        self.df = pd.read_csv(filepath)
        return self.df
