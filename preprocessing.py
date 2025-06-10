import pandas as pd
import csv

df = pd.read_csv('songs.csv')

df_new = df.assign(genre=df['genre'].str.split(', ')).explode('genre')

df_new['genre'] = df_new['genre'].str.strip()

df_new = df_new.reset_index(drop=True)

df_new['explicit'] = df_new['explicit'].astype(int)


df.to_csv('songs_clean.csv')

print(df_new['genre'].value_counts())

