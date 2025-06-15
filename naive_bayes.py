## Start with pre-processing of the csv files
import pandas as pd 
import numpy as np
from config import features
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import get_data, preprocess_data, transform_data
X_train, X_test, y_train, y_test, songs_train, songs_test= get_data()


all_genres = list(set(y_train))
comparable_attributes = features.copy()
comparable_attributes.remove('song')

num_bins = 20
genre_attribute_table = {
    genre: {
        attr: ([0]*num_bins) 
        for attr in comparable_attributes
    } 
    for genre in all_genres
}

# Determines range for bins
attribute_min_max = {}
for attr in comparable_attributes:
    idx = comparable_attributes.index(attr)
    min_val = X_test[:, idx].min()
    max_val = X_test[:, idx].max()
    attribute_min_max[attr] = (min_val, max_val)


bins = {}
for attr in comparable_attributes:
    min, max = attribute_min_max[attr]
    width = (max - min) / num_bins
    bins[attr] = [min + i * width for i in range(num_bins + 1)]

"""
Method to find the bin a score corresponds to
"""
def find_bin(val, boundaries):
    # For the binary attributes
    if len(boundaries) == 2:
        # Special case for the explicit attribute
        if val == 'FALSE':
            val = 0
        elif val == 'TRUE':
            val = 1
        if val == 0 or val == 1:
            return val
        else:
            raise ValueError(f"Expected binary value 0 or 1 for attribute, got {val}")
    if val < boundaries[0]:
        return 0
    for i in range(len(boundaries) - 1):
        if i == len(boundaries) - 2:
            if boundaries[i] <= val <= boundaries[i + 1]:
                return i
        else:
            if boundaries[i] <= val < boundaries[i + 1]:
                return i
    return len(boundaries) - 2

for i in range(len(X_train)):
    song = X_train[i]
    genre = y_train.iloc[i]

    for j, attr in enumerate(comparable_attributes):
        bin_idx = find_bin(song[j], bins[attr])
        genre_attribute_table[genre][attr][bin_idx] += 1

for genre in genre_attribute_table:
    for attr in genre_attribute_table[genre]:
        bin_counts = genre_attribute_table[genre][attr]
        total = sum(bin_counts)
        if total > 0:
            genre_attribute_table[genre][attr] = [round(count / total, 3) for count in bin_counts]
        else:
            genre_attribute_table[genre][attr] = [0 for _ in bin_counts]


"""
Method to predict the probabilities of each genre for a song
"""
def predict_genre_probabilities(song):
    genre_probs = {}

    for genre in genre_attribute_table:
        prob = 1.0
        for attr in comparable_attributes:
            idx = comparable_attributes.index(attr)
            attr_val = song[idx]
            bin_idx = find_bin(attr_val, bins[attr])
            bin_probs = genre_attribute_table[genre][attr]

            # Avoid multiplying by zero
            bin_idx = int(bin_idx)
            attr_prob = bin_probs[bin_idx] if bin_probs[bin_idx] > 0 else .000001
            prob *= attr_prob
        
        genre_probs[genre] = prob

    # Normalize to get probabilities
    total_prob = sum(genre_probs.values())
    if total_prob > 0:
        for genre in genre_probs:
            genre_probs[genre] = round(genre_probs[genre] / total_prob, 6)
    else:
        for genre in genre_probs:
            genre_probs[genre] = 0.0

    return genre_probs

def predict_multi_genres(song, threshold=0.1):
    probs = predict_genre_probabilities(song)
    filtered = [(genre, prob) for genre, prob in probs.items() if prob >= threshold]
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
    predicted_genres = [genre for genre, _ in filtered_sorted]
    return predicted_genres, dict(filtered_sorted)

"""
Method for plotting the distribution of a specific attribute for visualization purposes
"""
def plot_attribute_distribution(attr, genres_to_plot=None):
    if genres_to_plot is None:
        genres_to_plot = list(genre_attribute_table.keys())[:12]  # all
    
    plt.figure(figsize=(12, 8))

    for genre in genres_to_plot:
        bin_probs = genre_attribute_table[genre][attr]
        plt.plot(range(len(bin_probs)), bin_probs, marker='o', label=genre)
    
    plt.xticks(ticks=list(range(len(bin_probs))))
    plt.title(f"Distribution of attribute '{attr}' across genres")
    plt.xlabel("Bin Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
Method for plotting a heat map of attributes for a specific genre for visualization purposes
"""
def plot_genre_heatmap(genre):
    data = []
    for attr in comparable_attributes:
        data.append(genre_attribute_table[genre][attr])
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(data, cmap="YlGnBu", xticklabels=False, yticklabels=comparable_attributes)
    plt.title(f"Heatmap of bin probabilities for genre: {genre}")
    plt.xlabel("Bin Index")
    plt.ylabel("Attribute")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_attribute_distribution(comparable_attributes[0])
