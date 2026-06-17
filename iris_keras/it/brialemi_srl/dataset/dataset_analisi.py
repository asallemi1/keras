import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, normaltest

class DatasetAnalisi:

    def outliers_iqr_per_col(self, data):
        result = {}
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers_count = data[(data[col] < lower) | (data[col] > upper)].shape[0]
            result[col] = outliers_count
        return result

    def outliers_zscore_per_col(self, data, threshold=3):
        result = {}
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if col_data.std() == 0:
                result[col] = 0
                continue
            z_scores = (col_data - col_data.mean()) / col_data.std()
            outliers_count = np.sum(np.abs(z_scores) > threshold)
            result[col] = int(outliers_count)
        return result

    def valori_stringhe(self, data):
        cat_cols = data.select_dtypes(include=["object", "string"]).columns
        string = ""
        for col in cat_cols:
            string += f"\n{col} ({data[col].nunique()} valori unici): {data[col].unique()}"
        return string.strip()

    def valori_nulli(self, data):
        return data.isnull().sum().sort_values(ascending=False)

    def clean_data(self, data):
        return data.drop_duplicates()

    def correlazione(self, data):
        corr = data.corr(numeric_only=True)
        return corr.to_dict()

    def normality(self, data, alpha=0.05):
        numeric_cols = data.select_dtypes(include=np.number).columns
        risultati = []

        for col in numeric_cols:
            x = data[col].dropna()

            if len(x) < 8:
                risultati.append({
                    "variabile": col,
                    "n": len(x),
                    "test": None,
                    "statistica": np.nan,
                    "p_value": np.nan,
                    "normale": None,
                    "note": "Campione troppo piccolo"
                })
                continue

            if len(x) <= 5000:
                stat, p = shapiro(x)
                test = "Shapiro-Wilk"
            else:
                stat, p = normaltest(x)
                test = "D'Agostino-Pearson"

            risultati.append({
                "variabile": col,
                "n": len(x),
                "test": test,
                "statistica": float(round(stat, 4)),
                "p_value": float(round(p, 6)),
                "normale": bool(p > alpha),
                "note": ""
            })

        return risultati

    def pca(self, data, n_components=None, standardize=True):
        X = data.drop(columns=["Id", "Species"], errors="ignore")
        X = X.select_dtypes(include=np.number).copy()

        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_scaled)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=X.columns,
            columns=[f"PC{i + 1}" for i in range(pca.n_components_)]
        )

        explained_variance = pd.DataFrame({
            "Component": [f"PC{i + 1}" for i in range(pca.n_components_)],
            "Explained Variance": pca.explained_variance_ratio_,
            "Cumulative Variance": np.cumsum(pca.explained_variance_ratio_)
        })

        return {
            "n_components": int(pca.n_components_),
            "scores": scores.tolist(),
            "loadings": loadings.reset_index(names="Feature").to_dict(orient="records"),
            "explained_variance": explained_variance.to_dict(orient="records")
        }
