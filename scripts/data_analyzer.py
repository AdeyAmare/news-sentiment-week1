import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def check_data_quality(self):
        """Replicates original 'explore_data' functionality - Checks missing/dupes."""
        print("\n[ANALYZER] Checking Data Quality (Missing Values & Duplicates)...")
        print("-" * 30)
        
        # Missing
        missing = self.df.isnull().sum()
        missing_df = pd.DataFrame({'Missing': missing, '%': (missing/len(self.df))*100})
        print("Missing Values:\n", missing_df[missing_df['Missing'] > 0])
        
        # Duplicates
        dupes = self.df.duplicated().sum()
        print(f"Duplicate Rows: {dupes}")
        
        # Categorical Summary
        print("\nCategorical Summaries:")
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"{col}: {self.df[col].nunique()} unique. Top: {self.df[col].mode()[0]}")

    def get_exploratory_stats(self):
        """Replicates 'exploratory_analysis' - Basic stats summary."""
        print("\n[ANALYZER] Calculating Exploratory Statistics...")
        stats = {
            'shape': self.df.shape,
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'top_stocks': self.df['stock'].value_counts().head(10),
            'avg_headline_len': self.df['headline_length'].mean(),
            'unique_publishers': self.df['publisher'].nunique()
        }
        print(f"   -> Date Range: {stats['date_range']}")
        print(f"   -> Unique Publishers: {stats['unique_publishers']}")
        return stats

    def calculate_sentiment(self):
        """Calculates Sentiment polarity and categories."""
        print("\n[ANALYZER] Calculating Sentiment Scores (TextBlob)...")
        self.df['sentiment'] = self.df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        self.df['sentiment_category'] = pd.cut(
            self.df['sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # Print distribution
        dist = self.df['sentiment_category'].value_counts(normalize=True) * 100
        print("   -> Sentiment Distribution:")
        print(dist)
        return self.df

    def perform_topic_modeling(self, n_topics=5):
        """Runs LDA Topic Modeling."""
        print(f"\n[ANALYZER] Performing Topic Modeling (LDA, {n_topics} topics)...")
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(self.df['headline'])
        
        lda = LatentDirichletAllocation(
            n_components=n_topics, max_iter=10, 
            learning_method='online', random_state=42, n_jobs=-1
        )
        lda.fit(doc_term_matrix)
        
        print("   -> LDA Model fitted successfully.")
        return lda, vectorizer.get_feature_names_out()

    def extract_key_phrases(self):
        """Counts specific financial keywords."""
        print("\n[ANALYZER] Counting Key Financial Phrases...")
        financial_phrases = [
            'price target', 'upgrade', 'downgrade', 'earnings',
            'FDA approval', 'merger', 'acquisition', 'IPO',
            'stock split', 'dividend', 'guidance', 'analyst rating'
        ]
        
        phrase_counts = {}
        for phrase in financial_phrases:
            mask = self.df['headline'].str.lower().str.contains(phrase)
            phrase_counts[phrase] = mask.sum()
            
        phrase_df = pd.DataFrame.from_dict(phrase_counts, orient='index', columns=['count'])
        phrase_df['percentage'] = (phrase_df['count'] / len(self.df)) * 100
        return phrase_df.sort_values('count', ascending=True)

    def enrich_domain_data(self):
        """Extracts email domains from publishers."""
        print("\n[ANALYZER] Extracting Publisher Domains...")
        def extract_domain(publisher):
            if '@' in str(publisher):
                return publisher.split('@')[1]
            return publisher
            
        self.df['domain'] = self.df['publisher'].apply(extract_domain)
        return self.df