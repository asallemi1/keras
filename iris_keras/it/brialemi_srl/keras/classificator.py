from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
import numpy as np


class Keras:

    def __init__(self, data, use_pca=False):
        keras.utils.set_random_seed(42)
        self.val_model = None
        self.use_pca = use_pca
        self.model = None
        self.scaler = None
        self.pca = None
        self.label_encoder = None
        self.feature_columns = None
        self.history = None
        self.confusion_matrix = None
        self.set_mod(data)

    def set_mod(self, data):
        # Variabile target codificata numericamente: 0, 1, 2
        y = data["Species"]
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)

        # Variabili esplicative numeriche. Id e Species non devono entrare nel modello.
        X = data.drop(columns=["Id", "Species"], errors="ignore")
        X = X.select_dtypes(include=np.number)
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.use_pca:
            self.pca = PCA(n_components=2)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)

        model = keras.models.Sequential([
            keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

        self.model = model

        self.history = model.fit(
            X_train_scaled,
            y_train,
            epochs=200,
            batch_size=32,
            verbose=0
        )

        y_pred_prob = model.predict(X_test_scaled, verbose=0)
        y_pred_encoded = np.argmax(y_pred_prob, axis=1)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
        self.confusion_matrix = confusion_matrix(y_test, y_pred_encoded)

        score = model.evaluate(X_test_scaled, y_test, verbose=0)
        summary_buffer = StringIO()
        model.summary(print_fn=lambda line: summary_buffer.write(line + "\n"))

        self.val_model = {
            "classi": self.label_encoder.classes_.tolist(),
            "model_summary": summary_buffer.getvalue(),
            "y_test_numerico": y_test.tolist(),
            "predizioni_probabilita": y_pred_prob.tolist(),
            "predizioni_numeriche": y_pred_encoded.tolist(),
            "predizioni_labels": y_pred_labels.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "history": {
                key: [float(value) for value in values]
                for key, values in self.history.history.items()
            },
            "loss": score[0],
            "accuracy": score[1]
        }

    def get_val(self):
        return self.val_model

    def plot_training_history(self):
        history = self.history.history
        epochs = range(1, len(history["loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["loss"], marker="o")
        axes[0].set_title("Andamento loss")
        axes[0].set_xlabel("Epoca")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        axes[1].plot(epochs, history["accuracy"], marker="o")
        axes[1].set_title("Andamento accuracy")
        axes[1].set_xlabel("Epoca")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True)

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            ax=ax
        )
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predetto")
        ax.set_ylabel("Reale")
        plt.tight_layout()
        return fig

    def grafici(self):
        return {
            "training_history": self.plot_training_history(),
            "confusion_matrix": self.plot_confusion_matrix()
        }
