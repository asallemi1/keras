from iris_keras.it.brialemi_srl.dataset.dataset_manager import DatasetManager
from iris_keras.it.brialemi_srl.flask.flask_manager import FlaskManager


app = FlaskManager()
app.run(host='0.0.0.0', port=5000, debug=True)
            # con 0.0.0.0 consente di accedere tramite sia: http://127.0.0.1:5000 e http://192.168.1.228:5000
            # al primo posso accedere solo dalla macchina dove gira il programma
            # al secondo posso accedere solo tramite la LAN

print("Carico dataset")
ds_mg = DatasetManager()
print()

print("stampa dataset")
ds_mg.stampa()
print()

print("Esplorazione del dataset")
print(ds_mg.analisi())
print()

print("Visualizzazione grafici")
#ds_mg.grafici()
print()

print("pulisco dataset")
#ds_mg.clean()
print()

print("Analisi dati dopo il cleaning")
#print(ds_mg.analisi())
print()

print("stampa dataset dopo cleaning")
#ds_mg.stampa()
print()

print("correlazione")
#print(ds_mg.correlazione())
print()


