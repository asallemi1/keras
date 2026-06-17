import pandas as pd
from ucimlrepo import fetch_ucirepo
from Seeds.it.Brialemi_SRL.dataset.dataset_analisi import DatasetAnalisi
from Seeds.it.Brialemi_SRL.dataset.grafici import Grafici
from pathlib import Path

class DatasetManager: # è un classe CONTROLLER delle altre classi, non fa niente dal punto di vista di calcoli
    def __init__(self):

        self.__dftrain = self.load_file(
            columns=[
            "area",
            "perimetro",
            "compattezza",
            "lunghezza_kernel",
            "larghezza_kernel",
            "asimmetria",
            "lunghezza_solco", 
            "classe"]
        )
        #C:/Users/emanu/OneDrive/Documenti/GitHub/Kmeans/Seeds/seeds_dataset.txt
        #self.__dftest = self.load_file(
        #    "C:/Users/alisi/OneDrive/Documenti/GitHub/Kmeans/Seeds/seeds_test.txt",
        #    columns=[
        #    "classe"]
        #)

        self.__data_ana = DatasetAnalisi()
        self.__grafici = Grafici()
 

    def load_file(self, columns=None, sep=r"\s+"):
            return pd.read_csv(
                Path(__file__).resolve().parents[3] / "seeds_dataset.txt",
                sep=sep,
                header=None,
                names=columns,
                engine="python"
                )

    def set_data(self, data):
        self.__dataset = data

    def analisi(self): 
        val_nan = self.__data_ana.valori_nulli(self.__dftrain)
        val_strani = self.__data_ana.valori_stringhe(self.__dftrain)
        outliers = self.outlier()
        norm = self.__data_ana.normality(self.__dftrain)
        pca = self.__data_ana.pca(self.__dftrain) 
        return {
            "val_nan": val_nan,
            "val_strani": val_strani,
            "outliers": outliers,  # visto che sono tutti categorici fissi (opzioni) non ha senso parlare di outliers
            "test normalità": norm,
            "pca" : pca
        }

    def outlier(self):
        outl_iqr = self.__data_ana.outliers_iqr_per_col(self.__dftrain)
        outl_zscore = self.__data_ana.outliers_zscore_per_col(self.__dftrain)
        return {
            "outl_iqr": outl_iqr,
            "outl_zscore": outl_zscore
        }

    def grafici(self):              
        correlation = self.__grafici.plot_correlation(self.__dftrain) # impossibile fare su categorici
        pca = self.__data_ana.pca(self.__dftrain)
        grafico_pca = self.__grafici.plot_explained_variance(pca)
        grafico_kmeans = self.__grafici.plot_elbow(self.__dftrain)
        list_hist = []
        for col in self.__dftrain.columns:
            hist = self.__grafici.plot_hist(self.__dftrain, col)
            list_hist.append(hist)
        return {
            "correlation": correlation, 
            "hist": list_hist,
            "PCA" : grafico_pca,
            "grafico_kmeans" : grafico_kmeans
        }

    def clean(self):
        self.__dftrain = self.__data_ana.clean_data(self.__dftrain)

    def stampa(self):
        print(self.__dftrain)

    def correlazione(self):
        return self.__data_ana.correlazione(self.__dftrain)

    def get_datatrain(self):
        return self.__dftrain

   # def get_datatest(self):
    #    return self.__dftest

    def print_analisi(self):
        import json
        result = self.analisi()
        safe_result = self.make_json_safe(result)
        print(json.dumps(safe_result, indent=4, ensure_ascii=False))

    def make_json_safe(self, obj):
        import numpy as np
        import pandas as pd
        if isinstance(obj, dict):
            return {str(k): self.make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.make_json_safe(x) for x in obj]
        if isinstance(obj, pd.Series):
            return self.make_json_safe(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

