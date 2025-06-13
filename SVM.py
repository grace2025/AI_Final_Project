""" SVM """

from sklearn.svm import SVC
import pandas as pd
from preprocessing import transform_data, preprocess_data, get_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
import numpy as np


# ... should this be added to preprocessing ?
def create_genre_vectors(songs_train, songs_test, df, genres):
    
    df_all_song_genres = df.groupby('song')['genre'].apply(set).to_dict()
    
    y_train_vector_list = []
    for song in songs_train:
        song_genres = df_all_song_genres.get(song, set())
        true_genres = [1 if genre in song_genres else 0 for genre in genres]
        y_train_vector_list.append(true_genres)
    
    y_test_vector_list = []
    for song in songs_test:
        song_genres = df_all_song_genres.get(song, set())
        true_genres = [1 if genre in song_genres else 0 for genre in genres]
        y_test_vector_list.append(true_genres)
    
    return np.array(y_train_vector_list), np.array(y_test_vector_list)


def main():

    df = transform_data('songs.csv', 'genre')

    X_train, X_test, _, _, songs_train, songs_test = get_data()  

    unique_genres = df['genre'].unique().tolist()

    y_train, y_test = create_genre_vectors(
        songs_train=songs_train,
        songs_test=songs_test, 
        df=df,
        genres=unique_genres)


    model = MultiOutputClassifier(SVC(
        class_weight='balanced',
            kernel='rbf',
            gamma='scale',    # account for unbalanced data
        random_state=42,), n_jobs=-1) #n_jobs=-1 trains all SVMs in parallel


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # will need to change below metrics to custom metrics
    # accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy)

    classification_metric = classification_report(y_test, y_pred, target_names=unique_genres)
    print(classification_metric)



if __name__ == "__main__":
    main()