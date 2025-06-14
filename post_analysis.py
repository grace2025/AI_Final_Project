from config import features
from preprocessing import get_data
from naive_bayes import predict_multi_genres
import pandas as pd

df = pd.read_csv("songs.csv")
X_train, X_test, y_train, y_test, songs_train, songs_test= get_data()

for j in range(len(X_test)):
    print(f"{songs_test.iloc[j]} {predict_multi_genres(X_test[j])}, real genres: {df.loc[df['song'] == songs_test.iloc[j], 'genre'].values}")

