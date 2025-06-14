from config import features
from preprocessing import get_data
from naive_bayes import predict_multi_genres
import pandas as pd
import SVM

df = pd.read_csv("songs.csv")
X_train, X_test, y_train, y_test, songs_train, songs_test= get_data()

#for j in range(len(X_test)):
#    print(f"{songs_test.iloc[j]} {predict_multi_genres(X_test[j])}, real genres: {df.loc[df['song'] == songs_test.iloc[j], 'genre'].values}")

def get_real_genres_map():
    seen = set()
    song_to_genres = {}
    for j in range(len(X_test)):
        song = songs_test.iloc[j]
        if song in seen:
            continue
        seen.add(song)
        genres = df.loc[df['song'] == song, 'genre'].values
        if len(genres) > 0:
            genre_list = [g.strip() for g in genres[0].split(',')]
            song_to_genres[song] = set(genre_list)
    return song_to_genres

def score_classifier(classifier):
    real_genres_map = get_real_genres_map()
    svm_data = SVM.main()
    score = 0
    for i in range(len(X_test)):
        song = songs_test.iloc[i]
        if classifier == "NB":
            pred_genres = predict_multi_genres(X_test[i])[0]
        elif classifier == "SVM":
            pred_genres = svm_data[song]
        elif classifier == "NN":
            pred_genres = None
        real_genres = real_genres_map.get(song, set())
        score += get_score(pred_genres, real_genres, song)
    return score / len(X_test)

def get_score(pred_genres, real_genres, song):
    if len(pred_genres) == 0:
        return 0
    d = len(real_genres)
    n = 0
    for i in range(len(pred_genres)):
        if pred_genres[i] in real_genres:
            n += 1
    correct_score = n / d 
    d_incorrect = len(pred_genres)
    incorrect_score = n / d_incorrect
    return correct_score * incorrect_score  #If it only predicted genres in the real genres, incorrect_score = 1, 
                                            #ie. this penalizes for inccorect classifications

if __name__ == "__main__":
    print(f"Naive Bayes score: {score_classifier('NB')}")
    print(f"SVM score: {score_classifier('SVM')}")