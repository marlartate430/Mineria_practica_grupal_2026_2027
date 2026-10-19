import numpy as np
from typing import Tuple
from metrics import DistanceMetric, CosineMetric, EuclideanMetric
from abc import ABC
from kmeans_init import InitSignature, kmeans_plus_plus

class KMeans(ABC):
    def __init__(self, p_cluster_number: int, p_metric: DistanceMetric, p_init: InitSignature = kmeans_plus_plus, p_tol: float = 0.0, p_oversampling_factor: int | None = None, p_rounds: int | None = None, p_centroids: np.ndarray = None):
        """
        pre: Si p_centroids no es None, se da por hecho que se va a instanciar un modelo que ha sido cargado.
        """
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
        self.centroids: np.ndarray | None = p_centroids
        self.tol: float = p_tol
        self.init: InitSignature = p_init
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

    def save_model(self, file_path: str) -> None:
        if self.centroids is None:
            raise ValueError("El modelo no está entrenado. Llama a 'fit' primero.")
            
        estado = {
            "cluster_number": self.cluster_number,
            "tol": self.tol,
            "init": self.init,
            # JSON no entiende arrays de numpy, hay que convertirlos a listas
            "centroids": self.centroids.tolist(), 
            # Guardamos el nombre de la métrica para saber cuál instanciar al cargar
            "metric_name": self.metric.__class__.__name__ 
        }
        
        with open(file_path, 'w') as f:
            json.dump(estado, f, indent=4)

    @classmethod
    def load_model(cls, file_path: str):
        with open(file_path, 'r') as f:
            estado = json.load(f)
            
        # Reconstruir la métrica adecuada
        if estado["metric_name"] == "EuclideanMetric":
            metric = EuclideanMetric
        elif estado["metric_name"] == "CosineMetric":
            metric = CosineMetric
        else:
            # Fallback genérico o levantar error
            raise ValueError(f"Métrica {estado['metric_name']} desconocida")
            
        # Crear una nueva instancia de KMeans (o la subclase si se llama desde ella)
        modelo = cls(
            p_cluster_number=estado["cluster_number"],
            p_init=estado["init"],
            p_metric=metric,
            p_tol=estado["tol"]
            p_centroids=estado["centroids"]
        )
        
        # Restaurar los centroides como array de numpy
        modelo.centroids = np.array(estado["centroids"])
        return modelo

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
