from io import BytesIO
import base64

import pandas as pd
from flask import Flask, jsonify, request, render_template_string
import matplotlib.pyplot as plt

from Seeds.it.Brialemi_SRL.dataset.dataset_manager import DatasetManager
from Seeds.it.Brialemi_SRL.machine_learning.kmeans import Kmeans

class FlaskManager(object): # è una classe INTERFACCIA
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes() # inizializzazione delle varie root tutte insieme

        self.ds_mg = DatasetManager()

        self.ds_mg.clean()
        self.kmeans = Kmeans(self.ds_mg.get_datatrain(), n_clusters=3, use_pca=True)
        # in questo modo sia il cleaning sia i modelli vengono stimati in automatico

    def run(self, **kwargs):
        self.app.run(**kwargs) #**kwargs non specifico i parametri che dopo dovrò inserire

    def __register_routes(self): # __ indica un metodo che vede solo questa classe
        @self.app.route('/') # in automatico il metodo è GET 
        def index():
            return jsonify({
                'service': 'Kmeans API',
                'version': '1.0.0',
                'endpoints': {
                    '/datasetshow': 'GET - Head dataset',
                    '/info': 'GET - Statistiche descrittive',
                    '/grafici': 'GET - Grafici di correlazione, distribuzioni e PCA',
                    '/correlazione': 'GET - Matrice di correlazione',
                    '/valMod_kmeans': 'GET - Kmeans',
                    '/prevedi_kmeans': 'POST - Previsioni su file di test',
                    '/plot_kmeans': 'GET - Plot dei cluster'
                    },
            })

        @self.app.route('/datasetshow')
        def dataset_show():
            return jsonify(self.ds_mg.get_datatrain().head(5).to_dict()) # per ritornare un dataframe si usa .to_dict()

        @self.app.route('/info')
        def info():
            risp = self.ds_mg.analisi()

            def convert(obj):
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj

            risp = {k: convert(v) for k, v in risp.items()}

            return jsonify(risp)

        @self.app.route('/grafici')
        def grafici():
            grafici = self.ds_mg.grafici()
            figs = []
            for value in grafici.values(): # bisogna scomporre la scritta dall'immagine
                if value is None:
                    continue

                # Se è una lista di figure
                if isinstance(value, list): # infatti gli istogrammi sono più immagini in una lista
                    figs.extend(value) 
                else:
                    figs.append(value)

            images = []

            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format="png", bbox_inches="tight")
                img.seek(0)

                encoded = base64.b64encode(img.getvalue()).decode()
                images.append(encoded)

                plt.close(fig)

            html = """
               <h1>Plots</h1>
               {% for img in images %}
                   <img src="data:image/png;base64,{{ img }}" style="margin:10px;">
               {% endfor %}
               """

            return render_template_string(html, images=images) #prende il template html e usa su tutte le immagini 
                                                               #trasformate in formato base64

        @self.app.route('/correlazione')
        def correlazione():
            return jsonify(self.ds_mg.correlazione())

        
        @self.app.route('/valMod_kmeans')
        def valMod_kmeans():
            return jsonify(self.kmeans.get_val())

        @self.app.route('/prevedi_kmeans', methods=['POST'])
        def prevedi_kmeans():
            data = request.get_json()
            osservazione = data.get("osservazione")
            if osservazione is None:
                return jsonify({"error": "Nessuna osservazione fornita"}), 400

            try:
                predizione = self.kmeans.prevedi(osservazione)
                return jsonify({"predizione_cluster": predizione.tolist()})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            
    
        @self.app.route('/plot_kmeans')
        def plot_kmeans():

            img = self.kmeans.get_plot()

            html = """
            <h1>Grafico KMeans</h1>

            <img src="data:image/png;base64,{{ img }}"
            style="margin:10px;">
            """

            return render_template_string(
            html,
            img=img
        )
