# Iris Keras Classificator

API Flask per l'analisi del dataset Iris e per l'addestramento di un classificatore multiclasse realizzato con Keras.

L'applicazione carica il dataset Iris, prepara le variabili numeriche, codifica la variabile target `Species`, addestra una rete neurale feed-forward e rende disponibili tramite API metriche, predizioni e grafici.

## Dataset

Il progetto usa il file `Iris.csv`, composto da 150 osservazioni e 6 colonne:

- `Id`
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species`

La variabile target e' `Species`, con tre classi:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

Le feature usate dal modello sono le quattro variabili numeriche botaniche. La colonna `Id` e la target `Species` vengono escluse dalle feature di input.

## Preprocessing

Il preprocessing applicato al modello include:

- codifica numerica della target con `LabelEncoder`;
- split stratificato in training, validation e test set 70/15/15;
- standardizzazione delle feature con `StandardScaler`;
- fit dello scaler solo sul training set;
- transform dello scaler su validation e test set.

Lo split attuale e':

- training set: 105 osservazioni;
- validation set: 22 osservazioni;
- test set: 23 osservazioni.

## Modello Keras

Il classificatore e' un modello `Sequential` con layer densi:

```text
Input: 4 feature numeriche
Dense(16, activation="relu")
Dense(8, activation="relu")
Dense(3, activation="softmax")
```

Parametri principali:

- loss: `sparse_categorical_crossentropy`;
- optimizer: `Adam(learning_rate=0.001)`;
- metrica: `accuracy`;
- batch size: `32`;
- epochs: `200`. # ne bastano 100


## Endpoint API

Avviata l'applicazione, il servizio espone questi endpoint:

| Endpoint | Descrizione |
| --- | --- |
| `/` | Informazioni generali sul servizio e lista endpoint |
| `/datasetshow` | Prime righe del dataset |
| `/info` | Analisi descrittive, valori nulli, stringhe, outlier, normalita' e PCA |
| `/grafici` | Grafici di analisi del dataset |
| `/grafici_keras` | Grafici della history Keras e confusion matrix |
| `/correlazione` | Matrice di correlazione |
| `/valMod_keras` | Risultati del modello, summary, history, predizioni e metriche |

## Avvio locale

Creare e attivare un ambiente virtuale, poi installare le dipendenze:

```bash
pip install -r requirements.txt
```

Avviare l'API:

```bash
python main.py
```

L'applicazione espone Flask sulla porta `5000`:

```text
http://127.0.0.1:5000
```

Nel file `main.py` l'app viene avviata con:

```python
app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
```

`use_reloader=False` evita il doppio avvio del modello in modalita' debug.

## Docker

Il progetto include un `Dockerfile` per creare un container dell'applicazione.

Build locale:

```bash
docker build -t keras_classificator .
```

Run locale:

```bash
docker run -p 5000:5000 keras_classificator
```

## Docker Hub

L'applicazione e' stata pubblicata pubblicamente su Docker Hub:

```text
asallemi/keras_classificator
```

Per scaricare ed eseguire direttamente l'immagine pubblica:

```bash
docker pull asallemi/keras_classificator
docker run -p 5000:5000 asallemi/keras_classificator
```

Dopo l'avvio, l'API sara' disponibile su:

```text
http://127.0.0.1:5000
```

## Output del modello

L'endpoint `/valMod_keras` restituisce:

- classi del modello;
- summary Keras;
- dimensione di training, validation e test set;
- history di `loss`, `accuracy`, `val_loss` e `val_accuracy`;
- probabilita' predette;
- classi predette in formato numerico;
- classi predette come label originali;
- confusion matrix;
- loss e accuracy sul test set.

## Note

Il modello viene addestrato all'avvio dell'applicazione, quindi il primo avvio dell'API puo' richiedere qualche secondo.
