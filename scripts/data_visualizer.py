import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    def __init__(self, df):
        self.df = df
        sns.set_style("whitegrid")
        self.figsize = (12, 8)

    def plot_general_stats(self):
        """Replicates 'descriptive_statistics' visualizations."""
        print("\n[VISUALIZER] Plotting General Descriptive Stats...")
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Headline Length
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x='headline_length', bins=50)
        plt.title('Headline Length Distribution')
        
        # 2. Top Publishers
        plt.subplot(2, 2, 2)
        self.df['publisher'].value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Publishers')
        plt.xticks(rotation=45, ha='right')
        
        # 3. Day of Week
        plt.subplot(2, 2, 3)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.df['day_of_week'].value_counts().reindex(day_order).plot(kind='bar')
        plt.title('Articles by Day')
        
        # 4. Monthly Trend
        plt.subplot(2, 2, 4)
        self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month]).size().plot(kind='line')
        plt.title('Monthly Article Count')
        
        plt.tight_layout()
        plt.show()

    def plot_sentiment(self):
        """Replicates 'analyze_sentiment' plots."""
        print("\n[VISUALIZER] Plotting Sentiment Analysis...")
        # Bar Chart
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='sentiment_category', order=['Negative', 'Neutral', 'Positive'])
        plt.title('Sentiment Distribution')
        plt.show()
        
        # Time Series
        plt.figure(figsize=(12, 6))
        monthly = self.df.groupby(self.df['date'].dt.strftime('%Y-%m'))['sentiment'].mean()
        monthly.plot(kind='line')
        plt.title('Average Sentiment Over Time')
        plt.grid(True)
        plt.show()

    def plot_topics(self, lda_model, feature_names, n_topics=5):
        """Replicates 'extract_topics' plots."""
        print("\n[VISUALIZER] Plotting Topic Words...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for topic_idx, topic in enumerate(lda_model.components_):
            if topic_idx < n_topics:
                top_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_idx]
                top_weights = [topic[i] for i in top_idx]
                
                axes[topic_idx].barh(top_words, top_weights)
                axes[topic_idx].set_title(f'Topic {topic_idx + 1}')
                axes[topic_idx].invert_yaxis()
        
        plt.tight_layout()
        plt.show()

    def plot_key_phrases(self, phrase_df):
        """Replicates 'analyze_key_phrases' plot."""
        print("\n[VISUALIZER] Plotting Key Phrase Frequency...")
        plt.figure(figsize=(12, 6))
        phrase_df['count'].plot(kind='barh')
        plt.title('Financial Key Phrases Frequency')
        plt.tight_layout()
        plt.show()

    def plot_publication_patterns(self):
        """Replicates 'analyze_publication_patterns' plots."""
        print("\n[VISUALIZER] Plotting Publication Patterns...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Daily
        self.df.groupby(self.df['date'].dt.date).size().plot(ax=axes[0])
        axes[0].set_title('Daily Frequency')
        
        # Hourly
        self.df.groupby(self.df['date'].dt.hour).size().plot(kind='bar', ax=axes[1])
        axes[1].set_title('Hourly Distribution')
        
        # Day of Week
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.df.groupby(self.df['date'].dt.day_name()).size().reindex(dow_order).plot(kind='bar', ax=axes[2])
        axes[2].set_title('Day of Week Distribution')
        
        plt.tight_layout()
        plt.show()

    def plot_publisher_analysis(self):
        """Replicates 'analyze_publishers' plots."""
        print("\n[VISUALIZER] Plotting Publisher Analysis...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Top Publishers
        self.df['publisher'].value_counts().head(15).plot(kind='bar', ax=ax1)
        ax1.set_title('Top 15 Publishers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Top Domains (Assumes 'domain' column exists)
        self.df['domain'].value_counts().head(15).plot(kind='bar', ax=ax2)
        ax2.set_title('Top 15 Domains')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()