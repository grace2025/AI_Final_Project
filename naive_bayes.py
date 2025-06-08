## Start with pre-processing of the csv files
import pandas as pd 
import numpy as np
data = pd.read_csv("songs.csv")

attributes = data.columns.tolist()
data['genre_list'] = data['genre'].apply(lambda x: [g.strip() for g in x.split(',')])
all_genres = set(g for genres in data['genre_list'] for g in genres)

comparable_attributes = attributes
comparable_attributes.remove('artist')
comparable_attributes.remove('song')
comparable_attributes.remove('genre')
binary_attributes = {'explicit', 'mode'}

num_bins = 20
genre_attribute_table = {
    genre: {
        attr: ([0, 0] if attr in binary_attributes else [0]*num_bins) 
        for attr in comparable_attributes
    } 
    for genre in all_genres
}

attribute_min_max = {}
for attr in comparable_attributes:
    if attr not in binary_attributes:
        attribute_min_max[attr] = (data[attr].min(), data[attr].max())

bins = {}
for attr in comparable_attributes:
    if attr in binary_attributes:
        bins[attr] = [0, 1]
    else:
        min, max = attribute_min_max[attr]
        width = (max - min) / num_bins
        bins[attr] = [min + i * width for i in range(num_bins + 1)]

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

for _, song in data.iterrows():
    for genre in song['genre_list']:
        for attr in attributes:
            bin_idx = find_bin(song[attr], bins[attr])
            genre_attribute_table[genre][attr][bin_idx] += 1

for genre in genre_attribute_table:
    for attr in genre_attribute_table[genre]:
        bin_counts = genre_attribute_table[genre][attr]
        total = sum(bin_counts)
        if total > 0:
            genre_attribute_table[genre][attr] = [round(count / total, 3) for count in bin_counts]
        else:
            genre_attribute_table[genre][attr] = [0 for _ in bin_counts]


for genre in genre_attribute_table: 
    print(f"{genre}: {genre_attribute_table[genre]['explicit']}")

