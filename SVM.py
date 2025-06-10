""" SVM """

from sklearn.svm import SVC
import pandas as pd
from preprocessing import transform_data, preprocess_data
from sklearn.metrics import accuracy_score



df_clean = transform_data('songs.csv', 'genre')

features = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness', 
           'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
           'valence', 'tempo']

X = df_clean[features]
y = df_clean['genre']

X_train, X_test, y_train, y_test = preprocess_data(X, y)

model = SVC(
    class_weight='balanced',
    random_state=42,) # any other parameter necesssary?

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#feature importance? confusion matrix?