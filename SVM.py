""" SVM """

from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('songs_clean.csv')

features = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness', 
           'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
           'valence', 'tempo']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

print(df['genre'].value_counts())
