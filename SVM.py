# """ SVM """

from sklearn.svm import SVC
import pandas as pd
from preprocessing import transform_data, preprocess_data, get_data
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier



def main():

    X_train, X_test, y_train, y_test, songs_train, songs_test = get_data()  


    df = transform_data('songs.csv', 'genre')

    genre_counts = df['genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 2].index.tolist()
    removed_genres = genre_counts[genre_counts < 2].tolist()
    print(f'Removed genres: {removed_genres}')  

    model = MultiOutputClassifier(SVC(
        class_weight='balanced',
            kernel='rfr',
            gamma='scale',    # account for unbalanced data
        random_state=42,), n_jobs=-1) #n_jobs=-1 trains all SVMs in parallel


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

if __name__ == "__main__":
    main()