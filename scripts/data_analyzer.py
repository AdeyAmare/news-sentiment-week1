import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import spacy

nlp = spacy.load("en_core_web_sm")


class DataAnalyzer:
    def __init__(self, df):
        print("Initializing DataAnalyzer with dataframe...")
        self.df = df
        print(f"Initial dataframe loaded with {len(self.df)} rows and {len(self.df.columns)} columns.\n")

    def explore_data(self):
        print("\n================= STEP 1: EXPLORING RAW DATA =================")

        print("\n[1] Dataset Overview:")
        print("-" * 50)
        print(f"Total Records: {len(self.df)}")
        print(f"Columns: {', '.join(self.df.columns)}")

        print("\n[2] Checking Missing Values:")
        print("-" * 50)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        print(missing_df[missing_df['Missing Count'] > 0])

        print("\n[3] Checking Duplicate Rows:")
        print("-" * 50)
        duplicates = self.df.duplicated().sum()
        print(f"Duplicates Found: {duplicates} rows")

        if 'Unnamed: 0' in self.df.columns:
            duplicates_excl = self.df.drop('Unnamed: 0', axis=1).duplicated().sum()
            print(f"(Excluding 'Unnamed: 0') Duplicates: {duplicates_excl}")

        print("\n[4] Summary Statistics (Numeric Columns):")
        print("-" * 50)
        print(self.df.describe())

        print("\n[5] Categorical Column Analysis:")
        print("-" * 50)
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"\nColumn: {col}")
            print(f"Unique values: {self.df[col].nunique()}")
            print("Top 3 most common:")
            print(self.df[col].value_counts().head(3))

        print("\n=== Completed Data Exploration ===\n")
        return self.df

    def clean_data(self):
        print("\n================= STEP 2: CLEANING DATA =================")

        print("\n[1] Dropping rows missing headline, stock, or publisher...")
        before = len(self.df)
        self.df = self.df.dropna(subset=['headline', 'stock', 'publisher'])
        print(f"Removed {before - len(self.df)} rows.")

        print("\n[2] Removing duplicate headline/stock/date combinations...")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['headline', 'stock', 'date'], keep='first')
        print(f"Removed {before - len(self.df)} duplicates.")

        if 'Unnamed: 0' in self.df.columns:
            print("\n[3] Dropping unnecessary column 'Unnamed: 0'...")
            self.df = self.df.drop('Unnamed: 0', axis=1)

        print("\n[4] Fixing and parsing date column...")
        self.df['date'] = pd.to_datetime(self.df['date'], format='ISO8601', errors='coerce')
        before = len(self.df)
        self.df = self.df.dropna(subset=['date'])
        print(f"Removed {before - len(self.df)} rows with invalid dates.")

        print("\n[5] Cleaning text columns (headline, publisher, stock)...")
        self.df['headline'] = self.df['headline'].str.strip()
        self.df['headline'] = self.df['headline'].str.replace(r'\s+', ' ', regex=True)

        self.df['publisher'] = (
            self.df['publisher']
            .str.strip()
            .str.title()
            .str.replace(r'\s+', ' ', regex=True)
        )

        self.df['stock'] = (
            self.df['stock']
            .str.strip()
            .str.upper()
            .str.replace(r'[^\w\s]', '', regex=True)
        )

        print("\n[6] Cleaning URL column...")
        self.df['url'] = self.df['url'].str.strip().str.lower()

        print("\n[7] Removing remaining empty strings...")
        before = len(self.df)
        self.df = self.df.replace('', pd.NA).dropna()
        print(f"Removed {before - len(self.df)} rows.")

        print("\n[8] Sorting by date...")
        self.df = self.df.sort_values('date').reset_index(drop=True)

        print("\n[9] Creating new derived columns (year, month, day_of_week, headline_length, hour)...")
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['headline_length'] = self.df['headline'].str.len()
        self.df['hour'] = self.df['date'].dt.hour

        print("\n=== Completed Data Cleaning ===\n")
        return self.df

    def exploratory_analysis(self):
        print("\n================= STEP 3: DEEPER EXPLORATORY ANALYSIS =================")

        print("[1] Dataset Shape:")
        print(self.df.shape)

        print("\n[2] Date Range:")
        print(self.df['date'].min(), " --> ", self.df['date'].max())

        print("\n[3] Top 10 Most Covered Stocks:")
        print(self.df['stock'].value_counts().head(10))

        print("\n[4] Articles by Hour of Day:")
        print(self.df['hour'].value_counts().sort_index())

        print("\n[5] Headline Length Statistics:")
        print(self.df['headline_length'].describe())

        print("\n[6] Publisher Count:")
        print(f"Unique Publishers: {self.df['publisher'].nunique()}")

        print("\n[7] Top 5 Publishers:")
        print(self.df['publisher'].value_counts().head(5))

        print("\n[8] Articles by Day of Week:")
        print(self.df['day_of_week'].value_counts())

        print("\n=== Completed Exploratory Analysis ===\n")
        return self.df

    def analyze_sentiment(self):
        print("\n================= STEP 4: SENTIMENT ANALYSIS =================")

        print("[1] Calculating sentiment scores using VADER...")
        analyzer = SentimentIntensityAnalyzer()
        self.df['sentiment'] = self.df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

        print("[2] Categorizing sentiment into Negative / Neutral / Positive...")
        self.df['sentiment_category'] = pd.cut(
            self.df['sentiment'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['Negative', 'Neutral', 'Positive'],
        )

        print("\n[3] Sentiment Distribution:")
        sentiment_dist = self.df['sentiment_category'].value_counts()
        for cat, count in sentiment_dist.items():
            print(f"{cat}: {count} articles ({(count/len(self.df))*100:.1f}%)")

        print("\n=== Completed Sentiment Analysis ===\n")
        return self.df

    def extract_topics(self, num_topics=5, passes=2, chunksize=10000, no_below=10, no_above=0.3):
        print("\n================= STEP 5: TOPIC MODELING (Optimized for Single-Core) =================")
        print("[1] Preprocessing headlines for Gensim LDA...")

        # Tokenize headlines
        texts = [headline.lower().split() for headline in self.df['headline']]

        # Create dictionary and filter extremes
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        corpus = [dictionary.doc2bow(text) for text in texts]

        print(f"[2] Dictionary size: {len(dictionary)}")
        print(f"[3] Corpus size: {len(corpus)} documents")

        print(f"[4] Fitting LDA model with {num_topics} topics (single-core)...")
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            chunksize=chunksize,
            random_state=42
        )

        print("[5] Sample topics:")
        for idx, topic in lda.print_topics(-1):
            print(f"Topic {idx}: {topic}")

        print("\n=== Completed Topic Modeling ===\n")
        return lda, dictionary, corpus

    def analyze_key_phrases(self):
        print("\n================= STEP 6: KEY FINANCIAL PHRASE ANALYSIS =================")
        print("Searching headlines for common financial event keywords...")

        phrases = [
            'price target', 'upgrade', 'downgrade', 'earnings',
            'fda approval', 'merger', 'acquisition', 'ipo',
            'stock split', 'dividend', 'guidance', 'analyst rating'
        ]

        results = {}
        for phrase in phrases:
            count = self.df['headline'].str.lower().str.contains(phrase).sum()
            print(f"Phrase '{phrase}' found in {count} headlines.")
            results[phrase] = count

        print("\n=== Completed Key Phrase Analysis ===\n")
        return results

    def analyze_publishers(self):
        print("\n================= STEP 7: PUBLISHER & DOMAIN ANALYSIS =================")

        print("[1] Counting articles by publisher...")
        pub_counts = self.df['publisher'].value_counts()
        print(pub_counts.head(10))

        print("\n[2] Extracting domain names where publishers are emails...")

        def extract_domain(p):
            return p.split('@')[1] if '@' in str(p) else p

        self.df['domain'] = self.df['publisher'].apply(extract_domain)

        print("Domain extraction complete.")
        print(self.df['domain'].value_counts().head(10))

        print("\n=== Completed Publisher Analysis ===\n")
        return self.df

    def save_to_csv(self, output_path):
        print("\n================= STEP 8: SAVING CLEANED DATA =================")
        print(f"Saving cleaned dataset to: {output_path}")
        self.df.to_csv(output_path, index=False)
        print("Save complete!")
        print("\n=== All Processing Steps Finished ===\n")
        return self.df
