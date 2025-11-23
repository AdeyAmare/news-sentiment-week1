import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
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