import numpy as np
from abc import ABC, abstractmethod # Libreria que permite hacer clases abstractas y metodos abstractos
import distance
from typing import Tuple
from scipy.spatial.distance import cdist

## INICIALIZACION CON KMEANS++
"""
Para calcular las distancias utiliza la funcion
'euclidean(v1, v2) -> np.float64' que se encuentran en 'distance.py' 
"""
def kmeans_plus_plus(
    X: np.ndarray,
    k: int,
    random_state: int | None = None
) -> np.ndarray:
    """
    Inicialización de centroides mediante el algoritmo K-Means++.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de datos de tamaño (N_instancias, D_dimensiones).

    k : int
        Número de clusters que se quieren crear.

    random_state : int | None
        Semilla para reproducir la selección aleatoria de los centroides.

    Retorna
    -------
    np.ndarray
        Matriz de tamaño (k, D_dimensiones) con los centroides iniciales.
    """

    # Generador de números aleatorios
    rng = np.random.default_rng(random_state)

    # Número de instancias y dimensiones
    n_muestras, n_dimensiones = X.shape

    # Comprobación de los parámetros
    if k <= 0:
        raise ValueError(
            "El número de clusters k debe ser mayor que 0."
        )

    if k > n_muestras:
        raise ValueError(
            "El número de clusters k no puede ser mayor "
            "que el número de muestras."
        )

    # Matriz donde almacenaremos los centroides
    centroides = np.empty(
        (k, n_dimensiones),
        dtype=X.dtype
    )

    # ---------------------------------------------------------
    # 1. Seleccionar el primer centroide aleatoriamente
    # ---------------------------------------------------------

    primer_indice = rng.integers(0, n_muestras)

    centroides[0] = X[primer_indice]

    # ---------------------------------------------------------
    # 2. Calcular la distancia de cada punto al centroide
    #    seleccionado
    # ---------------------------------------------------------

    distancias_minimas = np.sum(
        (X - centroides[0]) ** 2,
        axis=1
    )

    # ---------------------------------------------------------
    # 3. Seleccionar el resto de centroides
    # ---------------------------------------------------------

    for i in range(1, k):

        # Suma de todas las distancias mínimas
        suma_distancias = np.sum(distancias_minimas)

        # Caso especial: todos los puntos coinciden
        if suma_distancias == 0:

            indice = rng.integers(0, n_muestras)

        else:

            # Probabilidad de seleccionar cada punto
            probabilidades = (
                distancias_minimas / suma_distancias
            )

            # Selección aleatoria según las probabilidades
            indice = rng.choice(
                n_muestras,
                p=probabilidades
            )

        # Guardar el nuevo centroide
        centroides[i] = X[indice]

        # -----------------------------------------------------
        # 4. Actualizar la distancia mínima de cada punto
        # -----------------------------------------------------

        nuevas_distancias = np.sum(
            (X - centroides[i]) ** 2,
            axis=1
        )

        distancias_minimas = np.minimum(
            distancias_minimas,
            nuevas_distancias
        )

    return centroides

"""
¿POR QUE CREAR CLASES ABSTRACTAS Y METODOS ABSTRACTOS?
Tenemos en mente utilizar diferentes implementaciones de kmeans.
- Una con KMeans normal (distancia euclidea)
- Otra con KMeans++ (distancia euclidea)
- Otra utilizando diferencia del coseno
- Otra utilizando Sentence Similarity
- Variantes que devuelven el mejor numero de clusters
"""

class IKMeans (ABC):
    """
    Interfaz abstracta para los modelos de clustering.
    Garantiza que cualquier algoritmo que implementemos sea compatible con 
    la estructura de evaluación.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Entrena el modelo de clustering.
        Precondición: X debe ser una matriz numérica (N_instancias, D_dimensiones).
        Postcondición: El modelo debe tener definidos sus centroides.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Asigna nuevos datos a los clusters existentes.
        Precondición: El modelo debe haber sido entrenado (fit ejecutado).
        Postcondición: Devuelve un array con las etiquetas de los clusters.
        """
        pass
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Método de conveniencia (este NO es abstracto, se hereda tal cual).
        Entrena y devuelve las etiquetas en un solo paso.
        """
        self.fit(X)
        return self.predict(X)

class KMeans(IKMeans):
    def __init__(self, p_cluster_number: int):
        self.cluster_number: int = p_cluster_number
        self.centroids: np.ndarray = None

    def fit(self, p_X: np.ndarray, p_threshold: float = 0.0) -> None:
        """
        ¿Como realizar la distancia?
        distance.euclidean(v1, v2)
        """

        N_instancias, D_dimensiones = X.shape
        
        # 1. Inicialización aleatoria de centroides (Puedes cambiarlo por K-Means++ luego)
        rng = np.random.default_rng()
        indices_aleatorios = rng.choice(N_instancias, size=self.n_clusters, replace=False)
        self.centroids = X[indices_aleatorios].copy()
        
        for i in range(self.max_iter):
            # --- FASE DE ASIGNACIÓN (Súper optimizada en C) ---
            # cdist calcula la distancia de todos los puntos a todos los centroides de golpe
            distancias = cdist(X, self.centroids, metric=self.metric)
            
            # argmin devuelve el índice del centroide más cercano para cada punto (axis=1)
            etiquetas = np.argmin(distancias, axis=1)
            
            # --- FASE DE ACTUALIZACIÓN ---
            nuevos_centroides = np.zeros((self.n_clusters, D_dimensiones))
            for k in range(self.n_clusters):
                # Filtramos los puntos asignados a este cluster 'k' (indexación booleana súper rápida)
                puntos_del_cluster = X[etiquetas == k]
                
                if len(puntos_del_cluster) > 0:
                    nuevos_centroides[k] = puntos_del_cluster.mean(axis=0)
                else:
                    # Si un cluster se queda vacío, le asignamos un punto aleatorio para que no colapse
                    nuevos_centroides[k] = X[rng.choice(N_instancias)]
                
            # --- COMPROBACIÓN DE CONVERGENCIA ---
            desplazamiento = np.linalg.norm(self.centroids - nuevos_centroides)
            self.centroids = nuevos_centroides
            
            if desplazamiento <= self.tol:
                print(f"[{self.metric.upper()}] Convergencia alcanzada en la iteración {i+1}")
                break
    
    def predict(self, p_X: np.ndarray) -> np.ndarray:
        if not self.centroids is None:
            pass
        else:
            raise "The model has not fitted yet. Call 'fit' method before calling 'predict'."
        
class CosineKMeans(IKMeans):
    def __init__(self, p_cluster_number: int):
        pass

    def fit(self, p_X: np.ndarray) -> None:
        """
        ¿Como realizar la distancia?
        distance.cosine_similarity(v1, v2)
        """
        pass
    
    def predict(self, p_X: np.ndarray) -> np.ndarray:
        pass

if __name__ == "__main__":
    print("Inicio")
    X: np.ndarray = np.random.randint(200, size=(10,8))
    print(X, end="\n\nAhora:")
    a: IKMeans = KMeans(5)
    a.fit(X)
