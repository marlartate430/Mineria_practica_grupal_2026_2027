from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
import numpy as np

# 1. Creamos el "contrato" que deben cumplir todas las métricas
class DistanceMetric(ABC):
    @staticmethod
    @abstractmethod
    def format_data(X: np.ndarray) -> np.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def clean_centroids(centroides: np.ndarray) -> np.ndarray:
        # if error < tol: returns None => the loop should end
        pass
        
    @staticmethod
    @abstractmethod
    def get_distance(X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def get_name():
        return "DistanceMetric"

# 2. Creamos la "mini clase" para Coseno
class CosineMetric(DistanceMetric):
    @staticmethod
    def format_data(X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        
    @staticmethod
    def clean_centroids(centroides: np.ndarray) -> np.ndarray:
        return CosineMetric.format_data(centroides) # Es la misma matemática
        
    @staticmethod
    def get_distance(X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        # cdist calcula la distancia de todos los puntos a todos los centroides de golpe
        return cdist(X, centroides, metric="cosine")

    @staticmethod
    def get_name():
        return "CosineMetric"

# 3. Creamos la "mini clase" para Euclídea
class EuclideanMetric(DistanceMetric):
    @staticmethod
    def format_data(X: np.ndarray) -> np.ndarray:
        return X.copy() # No hace falta hacer nada
        
    @staticmethod
    def clean_centroids(centroides: np.ndarray) -> np.ndarray:
        return centroides # No se normalizan
        
    @staticmethod
    def get_distance(X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        # cdist calcula la distancia de todos los puntos a todos los centroides de golpe
        return cdist(X, centroides, metric="sqeuclidean")

    @staticmethod
    def get_name():
        return "EuclideanMetric"
