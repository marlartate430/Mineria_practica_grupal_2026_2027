import numpy as np
from typing import Tuple
from metrics import DistanceMetric, CosineMetric, EuclideanMetric
from abc import ABC
from kmeans_init import InitSignature

class KMeans(ABC):
    def __init__(self, p_cluster_number: int, p_metric: DistanceMetric, p_init: str = "kmeans++", p_tol: float = 0.0, p_oversampling_factor: int | None = None, p_rounds: int | None = None):
        if p_cluster_number <= 0:
            raise ValueError(
                "El número de clusters k debe ser mayor que 0."
            )

        if p_init not in ("random", "kmeans++", "kmeans||"):
            raise ValueError(
                "Inicialización no válida."
            )

        if p_tol < 0:
            raise ValueError(
                "La tolerancia no puede ser negativa."
            )

        self.cluster_number: int = p_cluster_number
        self.centroids: np.ndarray | None = None
        self.tol: float = p_tol
        self.init: str = p_init
        self.metric: DistanceMetric = p_metric

    def fit(self, p_X: np.ndarray, p_seed: int | None = None) -> None:
        N_instancias, D_dimensiones = p_X.shape
        
        if self.cluster_number > N_instancias:
            raise ValueError(
                "El número de clusters k no puede ser mayor "
                "que el número de muestras."
            )

        X: np.ndarray = self.metric.format_data(p_X)

        # Hacer una clase que gestione las limpiezas internas y que pase por parámetros

        # 1. Inicialización aleatoria de centroides
        if self.init == "kmeans++":
            self._kmeans_plus_plus(X, p_seed)

        elif self.init == "kmeans||":

            if not isinstance(self.metric, EuclideanMetric):
                raise ValueError(
                    "K-Means|| clásico requiere EuclideanMetric."
                )

            self._kmeans_parallel(X, p_seed)

        elif self.init == "random":
            self._random_cluster_selection(X, p_seed)

        stop = False

        current_deviation_manager: Deviation_manager = Deviation_manager(self.cluster_number)

        while not stop:
            distancias = self.metric.get_distance(X, self.centroids)

            # argmin devuelve el índice del centroide más cercano para cada punto (axis=1)
            etiquetas = np.argmin(distancias, axis=1)
            
            current_deviation_manager.update(X, etiquetas, distancias)

            # --- FASE DE ACTUALIZACIÓN ---
            nuevos_centroides = np.zeros((self.cluster_number, D_dimensiones))
            for k in range(self.cluster_number):
                # Filtramos los puntos asignados a este cluster 'k' (indexación booleana súper rápida)
                puntos_del_cluster = X[etiquetas == k]
                
                if len(puntos_del_cluster) > 0:
                    nuevos_centroides[k] = puntos_del_cluster.mean(axis=0)
                else:
                    # ESTRATEGIA SCIKIT-LEARN: Coger el punto con mayor error del dataset
                    nuevos_centroides[k] = current_deviation_manager.get_most_deviated_instance_in_cluster()

            self.centroids = self.metric.clean_centroids(nuevos_centroides)

            # --- COMPROBACIÓN DE CONVERGENCIA ---
            desplazamiento = np.linalg.norm(self.centroids - nuevos_centroides) # Puedo usar metrics.get_distance()?
            self.centroids = nuevos_centroides
            
            if desplazamiento <= self.tol:
                stop = True
    
    def predict(self, p_X: np.ndarray) -> np.ndarray:
        result: np.ndarray = None

        if self.centroids is None:
            raise ValueError("The model has not fitted yet. Call 'fit' method before calling 'predict'.")
        elif self.centroids.shape[1] != p_X.shape[1]:
            raise ValueError("The centroids and the observation matrix do not have the same attributes number.")
        else:
            X: np.ndarray = self.metric.format_data(p_X)
            distances: np.ndarray = self.metric.get_distance(X, self.centroids)
            result = np.argmin(distances, axis=1)

        return result
    
class EuclideanKmeans(KMeans):
    def __init__(self, p_cluster_number: int, p_init: str, p_tol: float = 0.0):
        super().__init__(p_cluster_number, p_init, EuclideanMetric(), p_tol)

class SphericalKMeans(KMeans):
    def __init__(self, p_cluster_number: int, p_init: str, p_tol: float = 0.0):
        super().__init__(p_cluster_number, p_init, CosineMetric(), p_tol)

if __name__ == "__main__":
    print("Inicio")
    X: np.ndarray = np.random.randint(200, size=(10,8))
    print(X, end="\n\nAhora:")
    a: KMeans = KMeans(5)
    a.fit(X)
