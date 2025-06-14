""" SVM """

from sklearn.svm import SVC
import pandas as pd
from preprocessing import transform_data, preprocess_data, get_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def run_svm(report=False, return_data=True):

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
        random_state=42,), n_jobs=-1) # n_jobs=-1 trains all SVMs in parallel


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if report==True:
        classification_metric = classification_report(y_test, y_pred, target_names=unique_genres)
        print(classification_metric)

    if return_data==True:
        results = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'genres': unique_genres,
            'songs_test': songs_test,
            'X_test': X_test}
        
        pred_genres_dict = {}

        for i, song in enumerate(results['songs_test']):
            pred_genres = [results['genres'][j] for j, pred in enumerate(results['y_pred'][i]) if pred == 1]
            pred_genres_dict[song] = pred_genres
        
        return results, pred_genres_dict
    
def main():
    final_results = run_svm(report=False, return_data=True)
    return final_results[1]
    
if __name__ == "__main__":
    print(main())
    



        # could output best genre confusion matrix for each classifier? see how differently pop is classified across all songs and classifiers?
