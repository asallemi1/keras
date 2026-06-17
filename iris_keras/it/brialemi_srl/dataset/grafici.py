import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Grafici:
    def plot_correlation(self, data):
        corr = data.corr(numeric_only=True)
        fig = plt.figure(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di correlazione")
        #plt.show()
        return fig

    def plot_hist(self, data, col):
        fig = plt.figure(figsize=(6,4))

        plt.hist(
            data[col].dropna(),
            bins=30,
            edgecolor='black',
            linewidth=2)
        
        plt.title(f"Distribuzione di {col}")
        #plt.show()

        return fig

    def plot_distribution(self, data, col):
        fig = plt.figure(figsize=(6,4))
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribuzione di {col}")
        #plt.show()
        return fig

    def plot_scatter(self, data, x, y):
        fig = plt.figure(figsize=(6,4))
        sns.scatterplot(data=data, x=x, y=y)
        plt.title(f"{x} vs {y}")
        plt.show()
        return fig

    def plot_box(self, data, col):
        fig = plt.figure(figsize=(6,4))
        sns.boxplot(y=data[col])
        plt.title(f"Boxplot di {col}")
        plt.show()
        return fig

    def plot_counts(self, data, col):
        fig = plt.figure(figsize=(6,4))
        sns.countplot(y=data[col])
        plt.title(f"Distribuzione di {col}")
        plt.show()
        return fig
    
    def plot_explained_variance(self, pca_results):
        """
        Scree plot per la scelta del numero di componenti.
        """
        explained_variance = pca_results["explained_variance"]

        var_ratio = [
            row["Explained Variance"]
            for row in pca_results["explained_variance"]
        ]

        cum_var = np.cumsum(var_ratio)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            range(1, len(var_ratio) + 1),
            var_ratio,
            alpha=0.7,
            label="Varianza spiegata"
        )

        ax.plot(
            range(1, len(cum_var) + 1),
            cum_var,
            marker="o",
            label="Varianza cumulata"
        )

        ax.set_xlabel("Componenti principali")
        ax.set_ylabel("Proporzione di varianza")
        ax.set_title("Scree Plot PCA")

        ax.set_xticks(range(1, len(var_ratio) + 1))
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        #plt.show()

        return fig

    # ADESSO NON MOSTRO I GRAFICI IN LOCALE
