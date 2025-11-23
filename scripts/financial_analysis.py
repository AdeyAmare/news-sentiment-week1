import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class FinancialDataAnalyzer:
    def __init__(self, df):
        print("Initializing DataAnalyzer with dataframe...")
        self.df = df
        print(f"Initial dataframe loaded with {len(self.df)} rows and {len(self.df.columns)} columns.\n")

    def explore_data(self):
        print("\n================= STEP 1: EXPLORING RAW DATA (Descriptive Statistics) =================")

        print("\n[1] Dataset Overview:")
        print("-" * 50)
        print(f"Total Records: {len(self.df)}")
        print(f"Columns: {', '.join(self.df.columns)}")

        print("\n[2] Checking Missing Values:")
        print("-" * 50)
        missing = self.df.isnull().sum()
        print(missing[missing > 0])

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
        # This step is crucial for Time Series Analysis requirements
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

        # Creating derived columns for Descriptive Stats & Time Series Analysis
        print("\n[9] Creating new derived columns (year, month, day_of_week, headline_length, hour)...")
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['headline_length'] = self.df['headline'].str.len() # Requirement: Statistics for textual lengths
        self.df['hour'] = self.df['date'].dt.hour # Requirement: Analysis of publishing times

        print("\n=== Completed Data Cleaning ===\n")
        return self.df

    def exploratory_analysis(self):
        print("\n================= STEP 3: DEEPER EXPLORATORY ANALYSIS (Time Series & Publishing Times) =================")

        print("[1] Dataset Shape:")
        print(self.df.shape)

        print("\n[2] Date Range (Time Series Trends):")
        print(self.df['date'].min(), " --> ", self.df['date'].max())

        print("\n[3] Top 10 Most Covered Stocks:")
        print(self.df['stock'].value_counts().head(10))

        print("\n[4] Articles by Hour of Day (Publishing Times Analysis):")
        # Requirement: Analysis of publishing times
        print(self.df['hour'].value_counts().sort_index())

        print("\n[5] Headline Length Statistics (Descriptive Statistics):")
        # Requirement: Basic statistics for textual lengths
        print(self.df['headline_length'].describe())

        print("\n[6] Publisher Count:")
        print(f"Unique Publishers: {self.df['publisher'].nunique()}")

        print("\n[7] Top 5 Publishers (Descriptive Statistics):")
        # Requirement: Count articles per publisher
        print(self.df['publisher'].value_counts().head(5))

        print("\n[8] Articles by Day of Week (Time Series Trends):")
        print(self.df['day_of_week'].value_counts())

        print("\n=== Completed Exploratory Analysis ===\n")
        return self.df

    def extract_topics(self, num_topics=5):
        print("\n================= STEP 4: TEXT ANALYSIS (Topic Modeling with NLTK) =================")
        # Requirement: Extract topics or significant events
        
        print("[1] Preprocessing headlines (NLTK: Stopwords & Lemmatization)...")
        
        stop_words = set(stopwords.words('english'))
        # Add financial specific stop words to improve topic quality
        finance_stops = {'stock', 'market', 'stocks', 'shares', 'session', 'update', 'daily', 'mid-day', 'trading', 'today'}
        stop_words.update(finance_stops)
        
        lemmatizer = WordNetLemmatizer()

        def preprocess_text(text):
            # Remove non-alphabetic characters and lowercase
            text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            # Tokenize
            tokens = text.split()
            # Remove stopwords and short words, and lemmatize
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
            return " ".join(tokens)

        # Apply preprocessing
        # Using a sample for speed if dataset is huge, otherwise use full df
        sample_size = min(20000, len(self.df))
        print(f"Running topic modeling on a sample of {sample_size} headlines for performance...")
        text_data = self.df['headline'].sample(n=sample_size, random_state=42).apply(preprocess_text)

        print("[2] Vectorizing text (CountVectorizer)...")
        # Create Document-Term Matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(text_data)

        print(f"[3] Fitting LDA model with {num_topics} topics (scikit-learn)...")
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(dtm)

        print("\n[4] Discovered Topics:")
        feature_names = vectorizer.get_feature_names_out()
        
        topics_summary = {}
        for index, topic in enumerate(lda_model.components_):
            # Get top 10 words for each topic
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topic_str = f"Topic {index + 1}: {', '.join(top_words)}"
            print(topic_str)
            topics_summary[f"Topic {index+1}"] = top_words

        print("\n=== Completed Topic Modeling ===\n")
        return topics_summary

    def analyze_key_phrases(self):
        print("\n================= STEP 5: KEY FINANCIAL PHRASE ANALYSIS (Text Analysis) =================")
        # Requirement: Identify common keywords or phrases (significant events)
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
        print("\n================= STEP 6: PUBLISHER & DOMAIN ANALYSIS =================")

        print("[1] Counting articles by publisher (Publisher Analysis)...")
        # Requirement: Which publishers contribute most
        pub_counts = self.df['publisher'].value_counts()
        print(pub_counts.head(10))

        print("\n[2] Analyzing Publisher Content Focus (Publisher Analysis)...")
        # Requirement: Is there a difference in the type of news they report?
        
        top_publishers = self.df['publisher'].value_counts().head(5).index.tolist()
        
        def get_article_type(headline):
            headline = str(headline).lower()
            if 'earnings' in headline or 'eps' in headline or 'report' in headline: return 'Earnings/Reports'
            if 'upgrade' in headline or 'downgrade' in headline or 'rating' in headline or 'target' in headline: return 'Analyst Ratings'
            if 'fda' in headline or 'drug' in headline: return 'Pharma/Reg'
            if 'market' in headline or 'stocks' in headline: return 'General Market'
            return 'Other News'

        # Create a temporary column for this analysis
        self.df['article_type'] = self.df['headline'].apply(get_article_type)
        
        # Filter for top publishers to see the difference
        subset = self.df[self.df['publisher'].isin(top_publishers)]
        
        # Cross-tabulation to show the difference in news types
        cross_tab = pd.crosstab(subset['publisher'], subset['article_type'])
        print("\nPublisher Content Breakdown:")
        print(cross_tab)

        print("\n[3] Extracting domain names from email publishers (Publisher Analysis)...")
        # Requirement: If email addresses are used... identify unique domains
        
        def extract_domain(p):
            if isinstance(p, str) and '@' in p:
                return p.split('@')[1]
            return None

        domains = self.df['publisher'].apply(extract_domain).dropna()
        
        if not domains.empty:
            print("Unique domains found from email publishers:")
            print(domains.value_counts().head(10))
        else:
            print("No email address publishers found in the top counts.")

        print("\n=== Completed Publisher Analysis ===\n")
        return self.df

    def save_to_csv(self, output_path):
        print("\n================= STEP 7: SAVING CLEANED DATA =================")
        print(f"Saving cleaned dataset to: {output_path}")
        # Drop the temporary 'article_type' column before saving if you want to keep it clean
        if 'article_type' in self.df.columns:
            self.df = self.df.drop(columns=['article_type'])
            
        self.df.to_csv(output_path, index=False)
        print("Save complete!")
        print("\n=== All Processing Steps Finished ===\n")
        return self.df
    
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class FinancialDataVisualizer:
    def __init__(self, df):
        self.df = df

    def descriptive_statistics(self):
        """
        Visualizes descriptive statistics: headline lengths, top publishers, 
        weekly frequency, and monthly timeline.
        """
        fig = plt.figure(figsize=(15, 10))

        publisher_counts = self.df['publisher'].value_counts()
        day_counts = self.df['day_of_week'].value_counts()

        # 1. Headline Length Distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x='headline_length', bins=50, color='skyblue')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Length (Characters)')

        # 2. Top 10 Publishers
        plt.subplot(2, 2, 2)
        publisher_counts.head(10).plot(kind='bar', color='salmon')
        plt.title('Top 10 Publishers')
        plt.xticks(rotation=45, ha='right')

        # 3. Articles by Day of Week
        plt.subplot(2, 2, 3)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        # Ensure we only plot days that exist in the data
        existing_days = [d for d in day_order if d in day_counts.index]
        day_counts.reindex(existing_days).plot(kind='bar', color='lightgreen')
        plt.title('Articles by Day of Week')
        plt.xticks(rotation=45)

        # 4. Time Series: Monthly Article Count
        plt.subplot(2, 2, 4)
        # Group by year-month for a proper time series
        if 'date' in self.df.columns:
            monthly_counts = self.df.set_index('date').resample('M').size()
            monthly_counts.plot(kind='line', marker='o', color='purple')
            plt.title('Monthly Article Publication Frequency')
            plt.xlabel('Date')
            plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

    def plot_publisher_types(self):
        """
        Requirement: Publisher Analysis - Is there a difference in the type of news they report?
        Categorizes news and plots a stacked bar chart for top publishers.
        """
        print("Generating Publisher Content Type Analysis...")
        
        # 1. Define simple categorization logic
        def get_article_type(headline):
            headline = str(headline).lower()
            if 'earnings' in headline or 'eps' in headline or 'report' in headline or 'results' in headline: 
                return 'Earnings/Financials'
            if 'upgrade' in headline or 'downgrade' in headline or 'rating' in headline or 'target' in headline: 
                return 'Analyst Ratings'
            if 'fda' in headline or 'drug' in headline or 'trial' in headline: 
                return 'Pharma/Regulatory'
            if 'market' in headline or 'stocks' in headline or 'trade' in headline: 
                return 'General Market'
            if 'merger' in headline or 'acquisition' in headline or 'buy' in headline:
                return 'M&A'
            return 'Other News'

        # 2. Apply categorization locally for visualization
        temp_df = self.df.copy()
        temp_df['article_type'] = temp_df['headline'].apply(get_article_type)

        # 3. Get Top 5 Publishers
        top_publishers = temp_df['publisher'].value_counts().head(5).index.tolist()
        subset = temp_df[temp_df['publisher'].isin(top_publishers)]

        # 4. Create Crosstab
        cross_tab = pd.crosstab(subset['publisher'], subset['article_type'])

        # 5. Plot
        cross_tab.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
        plt.title('Content Focus of Top 5 Publishers')
        plt.ylabel('Number of Articles')
        plt.xlabel('Publisher')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='News Type')
        plt.tight_layout()
        plt.show()

    def visualize_topics(self, topics_summary):
        """
        Displays the topics extracted by the DataAnalyzer as a text chart.
        Args:
            topics_summary (dict): Dictionary {Topic X: [word1, word2...]}
        """
        if not topics_summary:
            print("No topics to visualize.")
            return

        n_topics = len(topics_summary)
        fig, axes = plt.subplots(1, n_topics, figsize=(15, 5))
        
        # Handle single axis if only 1 topic
        if n_topics == 1:
            axes = [axes]

        for idx, (topic_name, words) in enumerate(topics_summary.items()):
            ax = axes[idx]
            # Hide axes
            ax.axis('off')
            
            # Title
            ax.set_title(topic_name, fontsize=14, fontweight='bold', pad=10)
            
            # List words
            y_pos = 0.9
            for word in words:
                ax.text(0.5, y_pos, word, ha='center', va='center', fontsize=12)
                y_pos -= 0.08
        
        plt.suptitle("Key Topics Discovered (Top Keywords)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def visualize_key_phrases(self, phrase_counts):
        """
        Bar chart for specific financial phrases found.
        """
        if not phrase_counts:
            return

        phrase_df = pd.DataFrame.from_dict(phrase_counts, orient='index', columns=['count'])
        phrase_df = phrase_df.sort_values('count', ascending=True)

        plt.figure(figsize=(12, 6))
        phrase_df['count'].plot(kind='barh', color='teal')
        plt.title("Frequency of Specific Financial Events/Keywords")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.show()