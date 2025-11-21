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

    def visualize_topics(self, lda, dictionary, n_top_words=10):
        """
        Visualize topics from a Gensim LDA model.
        """
        topics = lda.show_topics(num_topics=-1, num_words=n_top_words, formatted=False)

        n_topics = len(topics)
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for idx, topic in topics:
            words, weights = zip(*topic)
            axes[idx].barh(words, weights)
            axes[idx].invert_yaxis()
            axes[idx].set_title(f"Topic {idx + 1}")

        # Remove empty subplots
        for ax in axes[n_topics:]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.show()

    def visualize_key_phrases(self, phrase_counts):
        phrase_df = pd.DataFrame.from_dict(phrase_counts, orient='index', columns=['count'])
        phrase_df = phrase_df.sort_values('count')

        phrase_df.plot(kind='barh', figsize=(12, 6))
        plt.title("Frequency of Financial Phrases")
        plt.show()
