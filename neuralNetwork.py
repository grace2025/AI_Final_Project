from preprocessing import get_data
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


class MusicGenreClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.history = None

    # ---------- Feature Extraction Helpers ----------
    def extract_mfcc(self, y, sr, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        return features

    def extract_spectral_features(self, y, sr):
        features = {}
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        return features

    def extract_rhythm_features(self, y, sr):
        features = {}
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        # Tempo Analysis
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        return features

    def extract_chroma_features(self, y, sr):
        features = {}
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(chroma.shape[0]):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        return features

    def extract_contrast_features(self, y, sr):
        features = {}
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features[f'contrast_{i}_mean'] = np.mean(contrast[i])
        return features

    def extract_tonnetz_features(self, y, sr):
        features = {}
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
        return features

    def extract_features(self, file_path, duration=30):
        try:
            y, sr = librosa.load(file_path, duration=duration)
            features = {}
            features.update(self.extract_mfcc(y, sr))
            features.update(self.extract_spectral_features(y, sr))
            features.update(self.extract_rhythm_features(y, sr))
            features.update(self.extract_chroma_features(y, sr))
            features.update(self.extract_contrast_features(y, sr))
            features.update(self.extract_tonnetz_features(y, sr))
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    # ---------- Model Building ----------
    def build_model(self, input_dim, num_classes):
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(input_dim,),
                               kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ---------- Training ----------
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=150):
        print("Starting training process...")

        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        y_test_enc = self.label_encoder.transform(y_test)

        # Set feature names for later use
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
        else:
            self.feature_columns = [f'feat_{i}' for i in range(X_train.shape[1])]

        print("\nDataset Info:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of genres: {len(self.label_encoder.classes_)}")
        print(f"Genres: {list(self.label_encoder.classes_)}")

        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print("Building model...")
        self.model = self.build_model(X_train_scaled.shape[1], len(self.label_encoder.classes_))
        print("\nModel Architecture:")
        self.model.summary()

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001, verbose=1
        )

        print(f"\nStarting training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train_scaled, y_train_enc,
            validation_data=(X_val_scaled, y_val_enc),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test_enc, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        y_pred = self.model.predict(X_test_scaled, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print("\nDetailed Classification Report:")
        print(classification_report(y_test_enc, y_pred_classes, target_names=list(self.label_encoder.classes_)))

        cm = confusion_matrix(y_test_enc, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Genre')
        plt.ylabel('True Genre')
        plt.tight_layout()
        plt.show()

        self.plot_training_history()
        return self.history

    # ---------- Training History Plot ----------
    def plot_training_history(self):
        if self.history is None:
            print("No training history available!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    # ---------- Prediction ----------
    def predict_genre(self, audio_file_path, top_n=3):
        if self.model is None:
            raise ValueError("The model is not trained yet. Call train() first.")

        print(f"Analyzing: {audio_file_path}")
        features = self.extract_features(audio_file_path)
        if features is None:
            return None

        # Convert features into DataFrame and align with expected feature order
        features_df = pd.DataFrame([features])
        missing_features = set(self.feature_columns) - set(features_df.columns)
        for col in missing_features:
            features_df[col] = 0
        features_df = features_df[self.feature_columns]

        features_scaled = self.scaler.transform(features_df)
        predictions = self.model.predict(features_scaled, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            genre = self.label_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx]
            results.append((genre, confidence))
        return results

    # ---------- Save and Load Model ----------
    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")

        dir_path = os.path.dirname(model_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        else:
            os.makedirs('.', exist_ok=True)

        self.model.save(f"{model_path}_model.h5")
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{model_path}_label_encoder.pkl")
        joblib.dump(self.feature_columns, f"{model_path}_features.pkl")
        print(f"Model and preprocessors saved to {model_path}")

    def load_model(self, model_path):
        try:
            self.model = keras.models.load_model(f"{model_path}_model.h5")
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            self.label_encoder = joblib.load(f"{model_path}_label_encoder.pkl")
            self.feature_columns = joblib.load(f"{model_path}_features.pkl")
            print(f"Model successfully loaded from {model_path}")
            print(f"Available genres: {list(self.label_encoder.classes_)}")
        except Exception as e:
            print(f"Error loading model: {e}")


def create_sample_dataset():
    sample_data = {
        'artist': ['The Beatles', 'Miles Davis', 'Bach', 'Nirvana', 'Daft Punk'],
        'title': ['Hey Jude', 'Kind of Blue', 'Brandenburg Concerto', 'Smells Like Teen Spirit', 'Around the World'],
        'genre': ['rock', 'jazz', 'classical', 'rock', 'electronic'],
        'file_path': [
            'audio/rock/beatles_hey_jude.wav',
            'audio/jazz/miles_davis_kind_of_blue.wav', 
            'audio/classical/bach_brandenburg.wav',
            'audio/rock/nirvana_smells_like.wav',
            'audio/electronic/daft_punk_around.wav'
        ]
    }
    return pd.DataFrame(sample_data)


def main():
    print("Loading and preprocessing data...")
    # Adjust filename if necessary; default is 'songs.csv'
    X_train, X_test, y_train, y_test, songs_train, songs_test = get_data('songs.csv')

    # Split validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    classifier = MusicGenreClassifier()
    history = classifier.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=150)

    # Save the trained model
    classifier.save_model("trained_models/music_genre_classifier")

    # Example prediction (replace with an actual file path)
    test_file = 'path/to/test_audio.wav'
    if os.path.exists(test_file):
        predictions = classifier.predict_genre(test_file)
        if predictions:
            print(f"\nPredictions for {test_file}:")
            for i, (genre, confidence) in enumerate(predictions, 1):
                print(f"  {i}. {genre}: {confidence:.3f} ({confidence * 100:.1f}%)")


def load_and_predict_example():
    """
    Example: load pre-trained model and make predictions.
    """
    classifier = MusicGenreClassifier()
    classifier.load_model('trained_models/music_genre_classifier')
    audio_file = 'path/to/new_song.wav'
    predictions = classifier.predict_genre(audio_file)
    if predictions:
        print(f"Predictions for {audio_file}:")
        for genre, confidence in predictions:
            print(f"  {genre}: {confidence:.3f} ({confidence * 100:.1f}%)")


if __name__ == "__main__":
    main()
