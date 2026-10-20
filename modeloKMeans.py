import json
from abc import ABC
import numpy as np
from typing import Tuple

from metrics import DistanceMetric, CosineMetric, EuclideanMetric
from kmeans_init import InitSignature, kmeans_plus_plus, random_cluster_selection, str_to_function_name, KMeans_parallel
from deviation_manager import Deviation_manager

class KMeans(ABC):
    def __init__(self, p_cluster_number: int, p_metric: DistanceMetric, p_init: InitSignature = kmeans_plus_plus, p_tol: float = 0.0, p_centroids: np.ndarray = None):
        """
        pre: Si p_centroids no es None, se da por hecho que se va a instanciar un modelo que ha sido cargado.
        """
        if p_cluster_number <= 0:
            raise ValueError(
                "El número de clusters k debe ser mayor que 0."
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

        self.inertia_: float | None = None # Se inicializa vacío

    def fit(self, p_X: np.ndarray, p_seed: int | None = None) -> None:
        N_instancias, D_dimensiones = p_X.shape
        
        if self.cluster_number > N_instancias:
            raise ValueError(
                "El número de clusters k no puede ser mayor "
                "que el número de muestras."
            )

        X: np.ndarray = self.metric.format_data(p_X)

        # Hacer una clase que gestione las limpiezas internas y que pase por parámetros

        self.centroids = self.init(p_X, self.cluster_number, self.metric, p_seed)

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

            nuevos_centroides = self.metric.clean_centroids(nuevos_centroides)

            # --- COMPROBACIÓN DE CONVERGENCIA ---
            distancias_movimiento = self.metric.get_distance(self.centroids, nuevos_centroides)
            desplazamiento = np.max(np.diag(distancias_movimiento))

            self.centroids = nuevos_centroides
            
            if desplazamiento <= self.tol:
                stop = True
        
        # Calculo de la inercia
        distancias_finales = self.metric.get_distance(X, self.centroids)
        distancias_minimas = np.min(distancias_finales, axis=1)
        self.inertia_ = float(np.sum(distancias_minimas))
    
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

    def save_model(self, file_path: str, p_class_name: str | None = None) -> None:
        if self.centroids is None:
            raise ValueError("El modelo no está entrenado. Llama a 'fit' primero.")
        
        class_name: str = self.__class__.__name__
        
        if not p_class_name is None:
            class_name = p_class_name

        estado = {
            "cluster_number": self.cluster_number,
            "tol": self.tol,
            "init_name": self.init.__name__ if hasattr(self.init, '__name__') else "KMeans_parallel",
            # JSON no entiende arrays de numpy, hay que convertirlos a listas
            "centroids": self.centroids.tolist(), 
            # Guardamos el nombre de la métrica para saber cuál instanciar al cargar
            "class_name": class_name
        }
        
        with open(file_path, 'w') as f:
            json.dump(estado, f, indent=4)

    @classmethod
    def load_model(cls, file_path: str):
        with open(file_path, 'r') as f:
            estado = json.load(f)
            
        clase = str2KMeansClass(estado["class_name"])
        init_obj: KMeans_parallel = None
        func_name: InitSiganture | KMeans_parallel = None

        if estado["init"] == "KMeans_parallel":
            init_obj = KMeans_parallel(estado["oversampling_factor"], estado["rounds"])
            func_name = init_obj.initialize
        else:
            func_name = str_to_function_name(estado["init"])

        # Crear una nueva instancia de KMeans (o la subclase si se llama desde ella)
        modelo = clase(
            p_cluster_number=estado["cluster_number"],
            p_init=estado["init"],
            p_tol=estado["tol"],
            p_centroids=np.array(estado["centroids"])
        )
        
        return modelo

class EuclideanKMeans(KMeans):
    def __init__(self, p_cluster_number: int, p_init: InitSignature, p_tol: float = 0.0, p_centroids: np.ndarray | None = None):
        super().__init__(p_cluster_number, EuclideanMetric, p_init, p_tol, p_centroids)

    def save_model(self, file_path: str) -> None:
        super().save_model(file_path, self.__class__.__name__)

class SphericalKMeans(KMeans):
    def __init__(self, p_cluster_number: int, p_init: InitSignature, p_tol: float = 0.0, p_centroids: np.ndarray | None = None):
        super().__init__(p_cluster_number, CosineMetric, p_init, p_tol, p_centroids)

    def save_model(self, file_path: str) -> None:
        super().save_model(file_path, self.__class__.__name__)

def str2KMeansClass(p_name: str):
    result = None
    # Reconstruir la clase adecuada
    if p_name == "EuclideanKMeans":
        result = EuclideanKMeans
    elif p_name == "SphericalKMeans":
        result = SphericalKMeans
    else:
        # Fallback genérico o levantar error
        raise ValueError(f"Class {p_name} desconocida")

    return result

if __name__ == "__main__":
    print("Inicio")
    X: np.ndarray = np.random.randint(200, size=(10,8))
    print(X, end="\n\nAhora:")
    a: KMeans = KMeans(5)
    a.fit(X)
