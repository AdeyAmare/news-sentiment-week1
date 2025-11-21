import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def descriptive_statistics(self):
        fig = plt.figure(figsize=(15, 10))

        publisher_counts = self.df['publisher'].value_counts()
        day_counts = self.df['day_of_week'].value_counts()

        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x='headline_length', bins=50)
        plt.title('Distribution of Headline Lengths')

        plt.subplot(2, 2, 2)
        publisher_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Publishers')

        plt.subplot(2, 2, 3)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day_counts.reindex(day_order).plot(kind='bar')
        plt.title('Articles by Day of Week')

        plt.subplot(2, 2, 4)
        monthly_counts = self.df.groupby([self.df['date'].dt.year, self.df['date'].dt.month]).size()
        monthly_counts.plot(kind='line')
        plt.title('Monthly Article Count')

        plt.tight_layout()
        plt.show()

    def plot_sentiment(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='sentiment_category',
                      order=['Negative', 'Neutral', 'Positive'])
        plt.title('Sentiment Distribution')
        plt.show()

        monthly = self.df.groupby(self.df['date'].dt.strftime('%Y-%m'))['sentiment'].mean()
        plt.figure(figsize=(12, 6))
        monthly.plot(kind='line')
        plt.title('Average Sentiment Over Time')
        plt.show()

    def visualize_topics(self, lda, vectorizer):
        feature_names = vectorizer.get_feature_names_out()
        n_top_words = 10

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_idx]
            top_vals = [topic[i] for i in top_idx]

            axes[idx].barh(top_words, top_vals)
            axes[idx].invert_yaxis()
            axes[idx].set_title(f"Topic {idx + 1}")

        plt.tight_layout()
        plt.show()

    def visualize_key_phrases(self, phrase_counts):

        phrase_df = pd.DataFrame.from_dict(phrase_counts, orient='index', columns=['count'])
        phrase_df = phrase_df.sort_values('count')

        phrase_df.plot(kind='barh', figsize=(12, 6))
        plt.title("Frequency of Financial Phrases")
        plt.show()
