from config import features
from preprocessing import get_data
from naive_bayes import predict_multi_genres
import pandas as pd
import SVM
from neural_network_updated import FromScratchNeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

df = pd.read_csv("songs.csv")
X_train, X_test, y_train, y_test, songs_train, songs_test, X_val, y_val, songs_val = get_data()

model_path = "trained_models/fromscratch_neural_network"

# Load parts
model_data = joblib.load(f"{model_path}_fromscratch_model.pkl")
scaler = joblib.load(f"{model_path}_scaler.pkl")
label_encoder = joblib.load(f"{model_path}_label_encoder.pkl")

# Reconstruct the model
nn = FromScratchNeuralNetwork(
    architecture=model_data['architecture'],
    learning_rate=model_data['learning_rate'],
    dropout_rate=model_data['dropout_rate']
)

# Restore weights, biases, feature names
nn.weights = model_data['weights']
nn.biases = model_data['biases']
nn.feature_columns = model_data['feature_columns']
nn.scaler = scaler
nn.label_encoder = label_encoder

nn_preds = nn.predict_flexible(X_test, method='adaptive', threshold=0.3)

svm_data = SVM.main()

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
    score = 0
    for i in range(len(X_test)):
        song = songs_test.iloc[i]
        if classifier == "NB":
            pred_genres = predict_multi_genres(X_test[i])[0]
        elif classifier == "SVM":
            pred_genres = svm_data[song]
        elif classifier == "NN":
            pred_genres = nn_preds[i]
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

def create_simple_visualization(classifiers):
    
    standard_scores = []
    at_least_one_scores = []
    
    for classifier in classifiers:
        standard_scores.append(score_classifier(classifier, 'standard'))
        at_least_one_scores.append(score_classifier(classifier, 'at least one'))
    
        
    # overall score barchart
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(classifiers, standard_scores, alpha=0.8, color='steelblue')
    
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('Standard Score Comparison')
    plt.ylim(0, 1)
    
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('overallscore_chart.png')
    plt.show()
    
    # At least one score barchart
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(classifiers, at_least_one_scores, alpha=0.8, color='lightcoral')
    
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('At Least One Score Comparison')
    plt.ylim(0, 1)
    
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('onescore_chart.png')
    plt.show()
 
    

if __name__ == "__main__":
    print(f"Naive Bayes standard score: {score_classifier('NB')}")
    print(f"SVM standard score: {score_classifier('SVM')}")
    print(f"NN standard score: {score_classifier('NN')}")
    print(f"Naive Bayes at least one score: {score_classifier('NB', 'at least one')}")
    print(f"SVM at least one score: {score_classifier('SVM', 'at least one')}")
    print(f"NN at least one score: {score_classifier('NN', 'at least one')}")
    
    classifiers = ['NB', 'SVM', 'NN']

    create_simple_visualization(classifiers)