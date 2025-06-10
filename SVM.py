""" SVM """

from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocessing import transform_data, preprocess_data


df_clean = transform_data('songs.csv', 'genre')


features = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness', 
           'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
           'valence', 'tempo']

X = df_clean[features]
y = df_clean['genre']

X_train, X_test, y_train, y_test = preprocess_data(X, y)

