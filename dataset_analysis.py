""" Initial dataset analysis """

import pandas as pd
import seaborn as sns
from preprocessing import transform_data
import matplotlib.pyplot as plt
import numpy as np



def plot_genre_counts(x, y, save=False):
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(y=y, x=x)
    plt.title('Genre Distribution', fontsize=20)
    plt.ylabel('Genre', fontsize=16)
    plt.xlabel('Number of Songs', fontsize=16)
    ax.bar_label(ax.containers[0], padding=3, fontsize=12)

    if save==True:
        plt.savefig('genre_counts.png')
        
    plt.show()

def main():

    df = transform_data('songs.csv', 'genre')
    genres = df['genre'].value_counts().keys().tolist()
    genre_counts = df['genre'].value_counts().values

    # plot_genre_counts(genre_counts, genres, save=False)

    # print(df.head(10))
    # print(df.shape)
    print(df.columns)


    genre_counts = df.groupby('genre').size()
    total_songs = len(df)
    genre_percentages = (genre_counts / total_songs) * 100

    # print(f"Pop: {genre_percentages['pop']}%")
    # print(f"Hip hop: {genre_percentages['hip hop']}%")
    # print(f"R&B: {genre_percentages['R&B']}%")
    # print(f"Dance/Electronic: {genre_percentages['Dance/Electronic']}%")

if __name__ == "__main__":
    main()

# classical was removed since it only had 1 song associated with it