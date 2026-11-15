import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class UnsupervisedAnalyzer:
    """
    Evaluación no supervisada del espacio de características mediante K-Means (k=2)
    y reducción de dimensionalidad con PCA (>= 95% varianza explicada).
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None

    def fit_kmeans(self, X: np.ndarray, n_clusters: int = 2):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        return cluster_labels, self.kmeans.cluster_centers_

    def fit_transform_pca(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Ajusta la proyección PCA evaluando la varianza acumulada.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.variance_threshold, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        if verbose:
            explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
            print(f"[PCA] Retenidas {self.pca.n_components_} componentes de {X.shape[1]} | Varianza explicada: {explained_var:.2f}%")
        return X_pca

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)