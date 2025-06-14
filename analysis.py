""" Initial dataset analysis """

import pandas as pd
import seaborn as sns
from preprocessing import transform_data
import matplotlib.pyplot as plt


def plot_genre_counts(x, y):
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(y=y, x=x)
    plt.title('Genre Distribution')
    plt.ylabel('Genre')
    plt.xlabel('Number of Songs')
    ax.bar_label(ax.containers[0], padding=3)

    plt.savefig('genre_counts.png')
    plt.show()

def main():

    df = transform_data('songs.csv', 'genre')
    genres = df['genre'].value_counts().keys().tolist()
    genre_counts = df['genre'].value_counts().values

    # plot_genre_counts(genre_counts, genres)

    print(df.head(10))

if __name__ == "__main__":
    main()

# classical was removed since it only had 1 song associated with it