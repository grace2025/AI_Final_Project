import pandas as pd
from config import features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def transform_data(filename, col):
    df = pd.read_csv(filename)

    df_new = df.assign(**{col: df[col].str.split(', ')}).explode(col)

    df_new[col] = df_new[col].str.strip()

    df_new = df_new.reset_index(drop=True)

    df_clean = df_new[df_new[col] != 'set()'].copy()

    # if value count is 1 or less, remove
    unqiue_genres = df_clean[col].value_counts()
    model_genres = unqiue_genres[unqiue_genres >= 2].index
    # removed_genres = unqiue_genres[unqiue_genres < 2].index
    # print(removed_genres)

    df_clean = df_clean[df_clean[col].isin(model_genres)]

    new_filename = filename.replace('.csv', '_clean.csv')

    df_clean.to_csv(new_filename, index=False)

    # print(df_clean['genre'].value_counts())

    return df_clean


def preprocess_data(X, y):
    songs = X['song'] 
    X_features = X.drop(columns='song')

    scaler = StandardScaler()
    scaler.fit(X_features)
    X_scaled = scaler.transform(X_features)

    X_train, X_test, y_train, y_test, songs_train, songs_test = train_test_split(
        X_scaled, y, songs, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val, songs_train, songs_val = train_test_split(
        X_test, y_test, songs_test, test_size=0.25, random_state=42
    )  # 60% train, 20% val, 20% test

    # return X_train, X_test, y_train, y_test, songs_train, songs_test
    return X_train, X_val, X_test, y_train, y_val, y_test, songs_train, songs_val, songs_test


def get_data():
    df_clean = transform_data('songs.csv', 'genre')

    df_clean['explicit'] = df_clean['explicit'].astype(int)
    df_clean['mode'] = df_clean['mode'].astype(int)

    df_clean['mode'] = df_clean['mode'].astype(int)



    # print(df_clean.columns)
    # print(df_clean['genre'].value_counts())

    X = df_clean[features]
    y = df_clean['genre']

    return preprocess_data(X, y)