# AI_Final_Project: Finding the Best Genre Prediction Model
### Anna Birge, Grace Bhagat, Zach Paquette, Hannah Storer

This is a Python project that compares different classifiers to determine the best classifier that predicts a songs genres. The data is pulled from Kaggle which originally pulled the data from Spotify.

Kaggle data: https://www.kaggle.com/code/varunsaikanuri/spotify-data-visualization

#### Classifiers: 
  - Neural Network
  - Naive Bayes
  - SVM
Neural Networks Classifier and the Naive Bayes Classifier are implemented from scratch. SVMs Classifier will use the sklearn model.


## About the Data
 - Shape: 3681, 18
 - 14 unqiue genres
```
Data columns (total 18 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   artist            2000 non-null   object 
 1   song              2000 non-null   object 
 2   duration_ms       2000 non-null   int64  
 3   explicit          2000 non-null   bool   
 4   year              2000 non-null   int64  
 5   popularity        2000 non-null   int64  
 6   danceability      2000 non-null   float64
 7   energy            2000 non-null   float64
 8   key               2000 non-null   int64  
 9   loudness          2000 non-null   float64
 10  mode              2000 non-null   int64  
 11  speechiness       2000 non-null   float64
 12  acousticness      2000 non-null   float64
 13  instrumentalness  2000 non-null   float64
 14  liveness          2000 non-null   float64
 15  valence           2000 non-null   float64
 16  tempo             2000 non-null   float64
 17  genre             2000 non-null   object 
```

## How to Run the Project
#### Step 1: Cone the Repository
```
git clone <repo_url>
cd <repo_name>
```

#### Step 2: Install Dependencies
Make sure you have Python 3.8+ installed. Then, install the dependencies listed in ```requirements.txt```
```
pip install -r requirements.txt
```

#### Step 3: Run ```preprocessing.py```
This script will pull the ```songs.csv``` file, horizontally transform the data to contain one genre in each row, remove genres with <1 counts and split the data to prep it as input for the models. This script outputs a ```songs_clean.csv``` file.
```
python preprocessing.py
```

#### Step 4: Run ```post_analysis.py```
This script calls each classifier and calculates an overall accuracy score allowing for consistent comparisions across classifiers. This script outputs two visualizations showing the outcomes of the three classifier's overall accuracy and accuracy in predicting at least one genre correctly.
```
python post_analysis.py
```


## Collaborators
The following individuals contributed to the development of this project:
1. Anna Birge
2. Zachary Parquette
3. Grace Bhagat
4. Hannah Storer
