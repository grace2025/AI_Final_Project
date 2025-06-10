import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('songs.csv')

df_new = df.assign(genre=df['genre'].str.split(', ')).explode('genre')

df_new['genre'] = df_new['genre'].str.strip()

df_new = df_new.reset_index(drop=True)

df_new['explicit'] = df_new['explicit'].astype(int)

df_clean = df_new[df_new['genre'] != 'set()'].copy()

# if value count is 1 or less, remove
unqiue_genres = df_clean['genre'].value_counts()
model_genres = unqiue_genres[unqiue_genres >= 2].index

df_clean = df_clean[df_clean['genre'].isin(model_genres)]

df_clean.to_csv('songs_clean.csv')

# print(df_clean['genre'].value_counts())


features = ['duration_ms', 'popularity', 'danceability', 'energy', 'loudness', 
           'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
           'valence', 'tempo']

print(df_clean.columns)
print(df_clean['genre'].value_counts())

X = df_clean[features]
y = df_clean['genre']

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')


