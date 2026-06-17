from numpy.f2py.crackfortran import verbose
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


class RegLogistica:

    def __init__(self, data):
        self.val_model = None
        self.set_mod(data)

    def set_mod(self, data):
        # Variabili esplicative e target
        y = data["Species"]
        num_classes = len(np.unique(y))

        # feature numeriche
        X = data.drop(columns=["Species"])

        self.feature_columns = X.columns.tolist()

        # Suddivisione train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        # Modello keras
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=len(X_train.columns),))
        model.add(keras.layers.Dense(6, activation="relu"))
        model.add(keras.layers.Dense(4, activation="relu"))
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        # Addestramento
        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=["accuracy", "precision", "recall", "f1_score"])

        self.model = model

        history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=32,
                            verbose=1)

        self.history = history

        y_pred = model.predict(X_test)

        score = model.evaluate(X_test, y_test, verbose=0)

        self.val_model = {
            "predizioni": y_pred.tolist(),
            "loss": score[0],
            "accuracy": score[1],
            "precision": score[2],
            "recall": score[3],
            "f1_score": score[4]
        }

    def get_val(self):
        return self.val_model


