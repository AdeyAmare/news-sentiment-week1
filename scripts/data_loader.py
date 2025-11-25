import pandas as pd


class DataLoader:
    """
    Handles reading tabular data from CSV files.

    Attributes
    ----------
    df : pandas.DataFrame or None
        Stores the most recently loaded dataset.
    """

    def __init__(self):
        self.df = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pandas.DataFrame
            The loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be parsed or is empty.
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        if df.empty:
            raise ValueError(f"CSV file is empty: {filepath}")

        self.df = df
        return df
