import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class FinancialDataAnalyzer:
    """
    A class for exploring, cleaning, analyzing, and performing NLP on financial news datasets.
    Methods include data exploration, cleaning, exploratory statistics, topic modeling, 
    key phrase analysis, publisher analysis, and CSV export.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a dataframe.
        Args:
            df (pd.DataFrame): Raw financial news data.
        """
        self.df = df.copy()
        print(f"[INIT] Loaded dataset with {len(df)} rows and {len(df.columns)} columns.\n")

    # ----------------------------------------------------------
    # 1. EXPLORE RAW DATA
    # ----------------------------------------------------------
    def explore_data(self) -> pd.DataFrame:
        """
        Explore basic characteristics of the dataset including missing values,
        duplicates, numeric and categorical summaries.
        Returns:
            pd.DataFrame: The original dataframe.
        """
        print("\n================ EXPLORING RAW DATA ================\n")
        print("[1] Dataset Overview")
        print(f"Rows: {len(self.df)} | Columns: {list(self.df.columns)}\n")
        
        print("[2] Missing Values")
        print(self.df.isnull().sum()[lambda x: x > 0], "\n")
        
        print("[3] Duplicate Rows")
        print(f"Total duplicates: {self.df.duplicated().sum()}\n")
        
        print("[4] Numeric Summary")
        print(self.df.describe(), "\n")
        
        print("[5] Categorical Columns Overview")
        for col in self.df.select_dtypes(include="object").columns:
            print(f"Column: {col}")
            print(f"- Unique: {self.df[col].nunique()}")
            print(self.df[col].value_counts().head(3), "\n")
        
        print("=== Completed Data Exploration ===\n")
        return self.df

    # ----------------------------------------------------------
    # 2. CLEAN DATA
    # ----------------------------------------------------------
    def clean_data(self) -> pd.DataFrame:
        """
        Clean dataset by removing missing values, duplicates, invalid dates,
        stripping text, and creating derived columns.
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        print("\n================ CLEANING DATA ================\n")

        # Drop rows with critical missing fields
        before = len(self.df)
        self.df = self.df.dropna(subset=["headline", "stock", "publisher"])
        print(f"Dropped {before - len(self.df)} rows with missing headline/stock/publisher.\n")

        # Drop duplicates
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=["headline", "stock", "date"])
        print(f"Removed {before - len(self.df)} duplicate rows.\n")

        # Drop unnecessary columns
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

        # Parse dates
        self.df["date"] = pd.to_datetime(self.df["date"], format='ISO8601', errors="coerce")
        before = len(self.df)
        self.df = self.df.dropna(subset=["date"])
        print(f"Removed {before - len(self.df)} rows with invalid dates.\n")

        # Clean text fields
        self.df["headline"] = self.df["headline"].str.strip().str.replace(r"\s+", " ", regex=True)
        self.df["publisher"] = self.df["publisher"].str.strip().str.title().str.replace(r"\s+", " ", regex=True)
        self.df["stock"] = self.df["stock"].str.strip().str.upper().str.replace(r"[^\w\s]", "", regex=True)
        self.df["url"] = self.df["url"].astype(str).str.strip().str.lower()

        # Remove empty strings
        before = len(self.df)
        self.df = self.df.replace("", pd.NA).dropna()
        print(f"Removed {before - len(self.df)} rows containing empty strings.\n")

        # Sort by date
        self.df = self.df.sort_values("date").reset_index(drop=True)

        # Derived columns
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day_of_week"] = self.df["date"].dt.day_name()
        self.df["headline_length"] = self.df["headline"].str.len()
        self.df["hour"] = self.df["date"].dt.hour

        print("=== Completed Data Cleaning ===\n")
        return self.df

    # ----------------------------------------------------------
    # 3. EXPLORATORY ANALYSIS
    # ----------------------------------------------------------
    def exploratory_analysis(self) -> pd.DataFrame:
        """
        Print statistics and distributions of stock articles, publisher counts,
        headline lengths, and article publishing times.
        Returns:
            pd.DataFrame: The dataframe with derived features.
        """
        print("\n================ EXPLORATORY ANALYSIS ================\n")
        print(f"Shape: {self.df.shape}\n")
        print(f"Date Range: {self.df['date'].min()} -> {self.df['date'].max()}\n")
        print("Top 10 Stocks:\n", self.df["stock"].value_counts().head(10), "\n")
        print("Articles by Hour:\n", self.df["hour"].value_counts().sort_index(), "\n")
        print("Headline Length Stats:\n", self.df["headline_length"].describe(), "\n")
        print("Publisher Count:", self.df["publisher"].nunique(), "\n")
        print("Top 5 Publishers:\n", self.df["publisher"].value_counts().head(5), "\n")
        print("Articles by Day:\n", self.df["day_of_week"].value_counts(), "\n")
        print("=== Completed Exploratory Analysis ===\n")
        return self.df

    # ----------------------------------------------------------
    # 4. TOPIC MODELING
    # ----------------------------------------------------------
    def extract_topics(self, num_topics: int = 5, sample_size: int = 50000) -> Dict[str, List[str]]:
        """
        Run LDA topic modeling on headlines.
        Args:
            num_topics (int): Number of topics to extract.
            sample_size (int): Maximum number of headlines to sample for performance.
        Returns:
            dict: Mapping from topic names to top words.
        """
        print("\n================ TOPIC MODELING ================\n")
        stop_words = set(stopwords.words("english"))
        finance_stops = {"stock", "market", "stocks", "shares", "session", "update", "daily", "trading", "today"}
        stop_words.update(finance_stops)

        lemmatizer = WordNetLemmatizer()

        def preprocess_text(text: str) -> str:
            text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
            tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
            return " ".join(tokens)

        sampled_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        text_data = sampled_df["headline"].apply(preprocess_text)

        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
        dtm = vectorizer.fit_transform(text_data)

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(dtm)

        feature_names = vectorizer.get_feature_names_out()
        topics_summary = {}
        for i, topic in enumerate(lda_model.components_):
            top_words = [feature_names[j] for j in topic.argsort()[-10:]]
            topics_summary[f"Topic {i + 1}"] = top_words

        print("=== Completed Topic Modeling ===\n")
        return topics_summary

    # ----------------------------------------------------------
    # 5. KEY PHRASE ANALYSIS
    # ----------------------------------------------------------
    def analyze_key_phrases(self) -> Dict[str, int]:
        """
        Count frequency of important financial keywords in headlines.
        Returns:
            dict: Mapping from keyword to occurrence count.
        """
        print("\n================ KEY PHRASE ANALYSIS ================\n")
        phrases = [
            "price target", "upgrade", "downgrade", "earnings",
            "fda approval", "merger", "acquisition", "ipo",
            "stock split", "dividend", "guidance", "analyst rating"
        ]
        counts = {p: self.df["headline"].str.lower().str.contains(p).sum() for p in phrases}

        for phrase, count in counts.items():
            print(f"{phrase}: {count}")
        print("=== Completed Key Phrase Analysis ===\n")
        return counts

    # ----------------------------------------------------------
    # 6. PUBLISHER ANALYSIS
    # ----------------------------------------------------------
    def analyze_publishers(self) -> pd.DataFrame:
        """
        Analyze publisher activity and classify article types.
        Returns:
            pd.DataFrame: Dataframe with article_type column added.
        """
        print("\n================ PUBLISHER ANALYSIS ================\n")
        print("Top Publishers:\n", self.df["publisher"].value_counts().head(10), "\n")

        def classify(headline: str) -> str:
            h = str(headline).lower()
            if any(w in h for w in ["earnings", "eps", "report"]): return "Earnings"
            if any(w in h for w in ["upgrade", "downgrade", "rating"]): return "Ratings"
            if any(w in h for w in ["fda", "drug"]): return "Pharma"
            if any(w in h for w in ["market", "stocks"]): return "Market"
            return "Other"

        self.df["article_type"] = self.df["headline"].apply(classify)

        top_pubs = self.df["publisher"].value_counts().head(5).index.tolist()
        cross_tab = pd.crosstab(
            self.df[self.df["publisher"].isin(top_pubs)]["publisher"],
            self.df[self.df["publisher"].isin(top_pubs)]["article_type"]
        )

        print("Publisher Breakdown:\n", cross_tab, "\n")
        print("=== Completed Publisher Analysis ===\n")
        return self.df

    # ----------------------------------------------------------
    # 7. SAVE
    # ----------------------------------------------------------
    def save_to_csv(self, path: str) -> pd.DataFrame:
        """
        Save cleaned dataframe to CSV.
        Args:
            path (str): File path for CSV.
        Returns:
            pd.DataFrame: Dataframe saved.
        """
        self.df.to_csv(path, index=False)
        print(f"Saved cleaned dataset to {path}")
        return self.df


class FinancialDataVisualizer:
    """
    Visualize financial data including descriptive stats, publisher distribution,
    topic modeling outputs, and key financial phrases.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def descriptive_statistics(self):
        """
        Plot distributions for headline length, top publishers, weekday counts, and monthly frequency.
        """
        fig = plt.figure(figsize=(15, 10))

        # Headline length
        plt.subplot(2, 2, 1)
        sns.histplot(self.df["headline_length"], bins=50)
        plt.title("Headline Length Distribution")

        # Top publishers
        plt.subplot(2, 2, 2)
        self.df["publisher"].value_counts().head(10).plot(kind="bar")
        plt.title("Top Publishers")
        plt.xticks(rotation=45)

        # Day of week
        plt.subplot(2, 2, 3)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.df["day_of_week"].value_counts().reindex(day_order).plot(kind="bar")
        plt.title("Articles by Day")

        # Monthly frequency
        plt.subplot(2, 2, 4)
        monthly = self.df.set_index("date").resample("M").size()
        monthly.plot(marker="o")
        plt.title("Monthly Article Count")

        plt.tight_layout()
        plt.show()

    def plot_publisher_types(self):
        """
        Stacked bar chart of article types by top 5 publishers.
        """
        df = self.df.copy()

        def classify(h):
            h = str(h).lower()
            if any(w in h for w in ["earnings", "report", "eps"]): return "Earnings"
            if any(w in h for w in ["upgrade", "downgrade", "target"]): return "Ratings"
            if any(w in h for w in ["fda", "drug"]): return "Pharma"
            if any(w in h for w in ["market", "stocks"]): return "Market"
            return "Other"

        df["article_type"] = df["headline"].apply(classify)
        top_pubs = df["publisher"].value_counts().head(5).index
        chart = pd.crosstab(df[df["publisher"].isin(top_pubs)]["publisher"],
                            df[df["publisher"].isin(top_pubs)]["article_type"])

        chart.plot(kind="bar", stacked=True, figsize=(12, 7))
        plt.title("Publisher Content Breakdown")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def visualize_topics(self, topics_summary: Optional[Dict[str, List[str]]] = None):
        """
        Visualize topic modeling results as horizontal bar charts.
        Args:
            topics_summary (dict): {"Topic 1": [word1, word2,...]}
        """
        if not topics_summary:
            print("No topics to visualize.")
            return

        for topic_name, words in topics_summary.items():
            plt.figure(figsize=(8, 4))
            plt.barh(words[::-1], range(1, len(words) + 1))
            plt.title(topic_name)
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

    def visualize_key_phrases(self, phrase_counts: Dict[str, int]):
        """
        Visualize key financial phrase counts as horizontal bar chart.
        Args:
            phrase_counts (dict): {"phrase": count}
        """
        phrase_df = pd.DataFrame.from_dict(phrase_counts, orient="index", columns=["count"])
        phrase_df.sort_values("count").plot(kind="barh", figsize=(10, 6))
        plt.title("Financial Phrase Frequency")
        plt.tight_layout()
        plt.show()
