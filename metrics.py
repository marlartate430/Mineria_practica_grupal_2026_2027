from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
import numpy as np

# 1. Creamos el "contrato" que deben cumplir todas las métricas
class DistanceMetric(ABC):
    @abstractmethod
    def format_data(self, X: np.ndarray) -> np.ndarray:
        pass
        
    @abstractmethod
    def clean_centroids(self, centroides: np.ndarray) -> np.ndarray:
        # if error < tol: returns None => the loop should end
        pass
        
    @abstractmethod
    def get_distance(self, X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        pass

# 2. Creamos la "mini clase" para Coseno
class CosineMetric(DistanceMetric):
    def format_data(self, X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        
    def clean_centroids(self, centroides: np.ndarray) -> np.ndarray:
        return self.format_data(centroides) # Es la misma matemática
        
    def get_distance(self, X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        # cdist calcula la distancia de todos los puntos a todos los centroides de golpe
        return cdist(X, centroides, metric="cosine")

# 3. Creamos la "mini clase" para Euclídea
class EuclideanMetric(DistanceMetric):
    def format_data(self, X: np.ndarray) -> np.ndarray:
        return X.copy() # No hace falta hacer nada
        
    def clean_centroids(self, centroides: np.ndarray) -> np.ndarray:
        return centroides # No se normalizan
        
    def get_distance(self, X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        # cdist calcula la distancia de todos los puntos a todos los centroides de golpe
        return cdist(X, centroides, metric="sqeuclidean")
