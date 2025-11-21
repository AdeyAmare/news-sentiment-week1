import pandas as pd
import os

class DataLoader:
    def __init__(self):
        self.df = None

    def load_data(self, filepath):
        """Loads data from CSV file."""
        print(f"\n[LOADER] Attempting to load data from: {filepath}")
        try:
            self.df = pd.read_csv(filepath)
            print(f"   -> Success! Loaded {len(self.df)} records.")
            return self.df
        except FileNotFoundError:
            print(f"   -> [ERROR] File not found at {filepath}")
            return None

    def clean_data(self, df):
        """
        Performs extensive data cleaning.
        Justification: Raw text data contains noise (HTML, extra spaces) and 
        invalid dates that will break analysis algorithms.
        """
        print("\n[LOADER] Starting data cleaning pipeline...")
        initial_count = len(df)

        # 1. Drop critical missing values
        df = df.dropna(subset=['headline', 'stock', 'publisher'])
        
        # 2. Remove duplicates
        df = df.drop_duplicates(subset=['headline', 'stock', 'date'], keep='first')
        
        # 3. Drop Unnamed columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # 4. Date Standardization
        print("   -> Standardizing dates...")
        df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')
        df = df.dropna(subset=['date'])

        # 5. Text Cleaning
        print("   -> Cleaning text fields (headlines, publishers, URLs)...")
        # Headline: Strip and remove extra whitespace
        df['headline'] = df['headline'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df['headline'] = df['headline'].str.replace(r'\s+', ' ', regex=True)
        
        # Publisher: Title case and strip
        df['publisher'] = df['publisher'].str.strip().str.title().str.replace(r'\s+', ' ', regex=True)
        
        # Stock: Uppercase and remove special chars
        df['stock'] = df['stock'].str.strip().str.upper().str.replace(r'[^\w\s]', '', regex=True)
        
        # URL: Lowercase
        df['url'] = df['url'].str.strip().str.lower()

        # 6. Remove empty strings that resulted from cleaning
        df = df.replace('', pd.NA).dropna()
        
        # 7. Sort and Reset
        df = df.sort_values('date').reset_index(drop=True)

        # 8. Feature Engineering (Derived Columns)
        print("   -> Generating derived features (Year, Month, Day, Length)...")
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.day_name()
        df['headline_length'] = df['headline'].str.len()
        df['hour'] = df['date'].dt.hour

        dropped_rows = initial_count - len(df)
        print(f"   -> Cleaning complete. Dropped {dropped_rows} rows. Final count: {len(df)}")
        return df

    def save_to_csv(self, df, output_path):
        """Saves the processed dataframe."""
        print(f"\n[LOADER] Saving processed data to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print("   -> Save successful.")