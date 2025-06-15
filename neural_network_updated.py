import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import joblib


class FromScratchNeuralNetwork:
    def __init__(self, architecture=[256, 128, 64], learning_rate=0.001, dropout_rate=0.2):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        print(f"Neural Network initialized with architecture: {architecture}")
        print(f"Learning rate: {learning_rate}, Dropout: {dropout_rate}")

    def initialize_weights(self, input_dim, num_classes):
        print(f"Initializing weights: {input_dim} -> {' -> '.join(map(str, self.architecture))} -> {num_classes}")
        layer_sizes = [input_dim] + self.architecture + [num_classes]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
        print(f"Initialized {len(self.weights)} weight matrices:")
        for i, w in enumerate(self.weights):
            print(f"Layer {i+1}: shape={w.shape}, mean={w.mean():.4f}, std={w.std():.4f}")

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(np.clip(x_shifted, -500, 500))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-15)

    def forward_pass(self, X, training=True):
        activations = [X]
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a = a * dropout_mask
            activations.append(a)
            current_input = a
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z_output)
        activations.append(output)
        return output, activations

    def backward_pass_with_weights(self, X, y, activations, sample_weights=None):
        m = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        # Compute error at output
        delta = activations[-1] - y  # shape: (m, num_classes)
        # Multiply each sample's error by its corresponding weight if provided
        if sample_weights is not None:
            delta = delta * sample_weights.reshape(-1, 1)
        # Backpropagation for all layers
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])
        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i]   -= self.learning_rate * bias_gradients[i]

    def compute_loss(self, predictions, y_true):
        m = y_true.shape[0]
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(predictions)) / m
        return loss

    def compute_accuracy(self, predictions, y_true):
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(pred_classes == true_classes)

    def to_one_hot(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def compute_class_weights(self, y_enc):
        # Compute weights inversely proportional to class frequencies
        classes, counts = np.unique(y_enc, return_counts=True)
        total = len(y_enc)
        class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
        return class_weights

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=200, batch_size=64, use_class_weights=True):
        print("Starting training process...")

        # Encode labels and convert to one-hot
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        y_test_enc = self.label_encoder.transform(y_test)
        num_classes = len(self.label_encoder.classes_)
        y_train_onehot = self.to_one_hot(y_train_enc, num_classes)
        y_val_onehot   = self.to_one_hot(y_val_enc, num_classes)
        y_test_onehot  = self.to_one_hot(y_test_enc, num_classes)

        # Store feature columns (assumed to come from a DataFrame)
        if isinstance(X_train, pd.DataFrame):
            self.feature_columns = X_train.columns.tolist()
            X_train = X_train.values
            X_val   = X_val.values
            X_test  = X_test.values
        else:
            self.feature_columns = [f'feat_{i}' for i in range(X_train.shape[1])]

        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled   = self.scaler.transform(X_val)
        X_test_scaled  = self.scaler.transform(X_test)

        # Initialize weights
        input_dim = X_train_scaled.shape[1]
        self.initialize_weights(input_dim, num_classes)

        # Compute class weights if desired
        if use_class_weights:
            class_weight_dict = self.compute_class_weights(y_train_enc)
            print("Class Weights:", class_weight_dict)
        else:
            class_weight_dict = None

        n_samples = X_train_scaled.shape[0]
        num_batches = int(np.ceil(n_samples / batch_size))

        # Training loop
        for epoch in range(epochs):
            # Shuffle the training data at each epoch
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train_scaled[permutation]
            y_shuffled = y_train_onehot[permutation]
            y_shuffled_labels = y_train_enc[permutation]
            epoch_loss = 0.0
            epoch_acc = 0.0

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                if use_class_weights:
                    # Compute sample weight for each sample in the batch
                    sample_weights = np.array([class_weight_dict[label] for label in y_shuffled_labels[start:end]])
                else:
                    sample_weights = None

                # Forward pass
                predictions, activations = self.forward_pass(X_batch, training=True)
                loss = self.compute_loss(predictions, y_batch)
                acc = self.compute_accuracy(predictions, y_batch)
                # Backward pass with sample weights applied
                weight_grads, bias_grads = self.backward_pass_with_weights(X_batch, y_batch, activations, sample_weights)
                self.update_weights(weight_grads, bias_grads)
                epoch_loss += loss * (end - start)
                epoch_acc += acc * (end - start)

            epoch_loss /= n_samples
            epoch_acc /= n_samples

            # Validation
            val_predictions, _ = self.forward_pass(X_val_scaled, training=False)
            val_loss = self.compute_loss(val_predictions, y_val_onehot)
            val_acc = self.compute_accuracy(val_predictions, y_val_onehot)

            self.training_history['loss'].append(epoch_loss)
            self.training_history['accuracy'].append(epoch_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Final evaluation on test set
        test_predictions, _ = self.forward_pass(X_test_scaled, training=False)
        test_accuracy = self.compute_accuracy(test_predictions, y_test_onehot)
        test_loss = self.compute_loss(test_predictions, y_test_onehot)
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Final Test Loss: {test_loss:.4f}")

        # Classification report and confusion matrix
        test_pred_classes = np.argmax(test_predictions, axis=1)
        print("\nClassification report:")
        print("-" * 50)
        unique_test_classes = np.unique(y_test_enc)
        unique_pred_classes = np.unique(test_pred_classes)
        all_classes = np.unique(np.concatenate([unique_test_classes, unique_pred_classes]))
        target_names = [self.label_encoder.classes_[i] for i in all_classes]
        print(f"Classes in test set: {len(unique_test_classes)}")
        print(f"Classes in predictions: {len(unique_pred_classes)}")
        print(f"Total classes in training: {len(self.label_encoder.classes_)}")
        print(classification_report(y_test_enc, test_pred_classes, 
                                  labels=all_classes,
                                  target_names=target_names,
                                  zero_division=0))
        print("\nGenerating confusion matrix...")
        cm = confusion_matrix(y_test_enc, test_pred_classes, labels=all_classes)
        plt.figure(figsize=(12, 10))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix - From-Scratch Neural Network')
        plt.xlabel('Predicted Genre')
        plt.ylabel('True Genre')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        self.plot_training_history()
        return self.training_history

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Plot training and validation accuracy
        ax1.plot(self.training_history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.training_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Neural Network Accuracy', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Plot training and validation loss
        ax2.plot(self.training_history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Neural Network Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        predictions, _ = self.forward_pass(X_scaled, training=False)
        pred_classes = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(pred_classes)
        return predicted_labels

    def save_model(self, model_path):
        import os
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, f"{model_path}_fromscratch_model.pkl")
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{model_path}_label_encoder.pkl")
        print(f"From-scratch model saved to {model_path}")


def main():
    print("Neural Network (From Scratch)")
    print("=" * 70)
    print("No TensorFlow, PyTorch, or scikit‑learn neural networks!")
    print("Everything implemented from scratch using only NumPy\n")
    print("Loading data...")
    X_train, X_test, y_train, y_test, songs_train, songs_test = get_data()
    print("Creating validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print("Initializing neural network...")
    nn = FromScratchNeuralNetwork(
        architecture=[256, 128, 64],
        learning_rate=0.005,
        dropout_rate=0.1
    )

    print("Training network...")
    nn.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=200, batch_size=64, use_class_weights=True)

    print("Saving model...")
    nn.save_model("trained_models/fromscratch_neural_network")

    print("Making sample predictions...")
    # Use first 10 samples for illustration
    sample_preds = nn.predict(pd.DataFrame(X_test[:10], columns=nn.feature_columns))
    print("\nSample predictions:")
    print("-" * 30)
    for i, pred in enumerate(sample_preds, 1):
        if songs_test is not None and len(songs_test) >= 10:
            songs_list = songs_test.tolist() if hasattr(songs_test, 'tolist') else list(songs_test)
            print(f"{i:2d}. {songs_list[i-1]}: Predicted = {pred}")
        else:
            print(f"{i:2d}. Predicted = {pred}")


if __name__ == "__main__":
    main()