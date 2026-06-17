from numpy.f2py.crackfortran import verbose
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from iris_keras.it.Brialemi_SRL.dataset.dataset_analisi import DatasetAnalisi


class Keras:

    def __init__(self, data, use_pca=False):
        self.val_model = None
        self.set_mod(data)
        self.use_pca = use_pca
        self.model = None
        self.scaler = None

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

        # Standardizzazione se non usa PCA, altrimenti la PCA include già la standardizzazione
        if not self.use_pca:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
        else:
            # faimo PCA con 3 componenti principali (o un numero a scelta) e standardizziamo i dati prima di applicare KMeans
            data_ana = DatasetAnalisi()
            pca_results = data_ana.pca(X, n_components=2, standardize=True)
            X_scaled = pca_results["scores"]
            self.X_pca = np.array(X_scaled)

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

        history = model.fit(X_scaled, y_train,
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


