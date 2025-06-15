## used claude ai for help with idea generation for this algorithm, troubleshooting, 
## and to learn better how to implement a neural network AI algorithm.
## we should change this more to fit our dataset, this is just a baseline and should be edited more before use!

## Basic Usage Summary:

"""
# Load data from CSV using the preprocessing module
from preprocessing import get_data
# This returns training and test data (features, labels, and song identifiers)
X_train, X_test, y_train, y_test, songs_train, songs_test = get_data()

# Initialize classifier and split a validation set from the training data if needed
from neuralNetwork import MusicGenreClassifier
classifier = MusicGenreClassifier()

# (Optional) Split out validation data from X_train/y_train if not already done in get_data()
# For example:
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Train the model (pass your training, validation, and test sets)
classifier.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=150)

# Save the trained model and its pre-processors (scaler, label encoder, etc.)
classifier.save_model('my_genre_classifier')

## To make predictions:

# Load the trained model
classifier = MusicGenreClassifier()
classifier.load_model('my_genre_classifier')

# Predict genres on test data using table features
predicted_labels = classifier.predict_from_table(X_test)
for song, actual, pred in zip(songs_test, y_test, predicted_labels):
    print(f"{song}: Actual = {actual}, Predicted = {pred}")

"""


## Neural Network code 
from preprocessing import get_data
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras  # using this style to avoid conflicts
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
        
        # Optimizer with gradient clipping helps with stability
        optimizer = keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
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

        # Store feature columns (assumed to be from the DataFrame)
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

        # Compute class weights to address imbalance
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        classes = np.unique(y_train_enc)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_enc)
        class_weight_dict = dict(zip(classes, class_weights))
        print("\nClass Weights:", class_weight_dict)

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
            class_weight=class_weight_dict,  # Using class weights here
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
        all_genres = list(set(y_train))
        all_labels = self.label_encoder.transform(all_genres)
        print(classification_report(y_test_enc, y_pred_classes, labels=all_labels,
                                    target_names=list(self.label_encoder.classes_)))

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

    # ---------- Prediction from Table Data ----------
    def predict_from_table(self, X):
        """
        Expects a DataFrame (or array-like object) X with the same feature columns as during training.
        Returns an array of predicted genre labels.
        """
        # Ensure X is a DataFrame with the correct column order
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        else:
            X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        pred_indices = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(pred_indices)
        return predicted_labels

    # ---------- Save Predictions from Test Set to CSV ----------
    def save_predictions_from_test_set(self, output_csv, X_test, y_test, songs_test=None):
        """
        Given test features (X_test), true labels (y_test), and optionally song identifiers (songs_test),
        predict genre for each sample and write a CSV 
        """
        # Reconstruct a DataFrame for X_test if needed
        if not isinstance(X_test, pd.DataFrame):
            df_test = pd.DataFrame(X_test, columns=self.feature_columns)
        else:
            df_test = X_test.copy()
        df_test['true_genre'] = y_test
        predicted_labels = self.predict_from_table(df_test)
        df_test['predicted_genre'] = predicted_labels
        if songs_test is not None:
            df_test['song'] = songs_test
        df_test.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

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


def main():
    print("Loading and preprocessing data...")
    # get_data() reads and preprocesses songs.csv and returns table data and identifiers.
    X_train, X_test, y_train, y_test, songs_train, songs_test = get_data()

    # Split a validation set from the training data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      stratify=y_train,
                                                      random_state=42)

    classifier = MusicGenreClassifier()
    print("\nTraining the model...")
    classifier.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=150)

    # Save the trained model and associated pre-processors
    classifier.save_model("trained_models/music_genre_classifier")

    # Generate and print a few sample predictions from the test set.
    sample_preds = classifier.predict_from_table(pd.DataFrame(X_test, columns=classifier.feature_columns))
    print("\nSample predictions:")
    for i, pred in enumerate(sample_preds, 1):
        if songs_test is not None:
            print(f"{i}. {songs_test[i-1]}: Predicted = {pred}")
        else:
            print(f"{i}. Predicted = {pred}")

    # Save all test predictions to a CSV file
    classifier.save_predictions_from_test_set("songs_predictions.csv", X_test, y_test, songs_test)


if __name__ == "__main__":
    main()
