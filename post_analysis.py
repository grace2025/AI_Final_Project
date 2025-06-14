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

def score_classifier(classifier, score_type= 'standard'):
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
        if score_type == "standard":
            score += get_score_standard(pred_genres, real_genres, song)
        elif score_type == "at least one":
            score += get_score_at_lest_one(pred_genres, real_genres)

    return round(score / len(X_test), 3)

def get_score_standard(pred_genres, real_genres, song):
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

def get_score_at_lest_one(pred_genres, real_genres):
    for i in range(len(pred_genres)):
        if pred_genres[i] in real_genres:
            return 1
    return 0
    

if __name__ == "__main__":
    print(f"Naive Bayes standard score: {score_classifier('NB')}")
    print(f"SVM standard score: {score_classifier('SVM')}")
    print(f"Naive Bayes at least one score: {score_classifier('NB', 'at least one')}")
    print(f"SVM at least one score: {score_classifier('SVM', 'at least one')}")
    