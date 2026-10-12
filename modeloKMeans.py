import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist

class KMeans:

    real_distance_metrics = {
            "euclidean" : "sqeuclidean", # sqeuclidean(v1, v2) = dot(v1-v2, v1-v2)
            "cosine" : "cosine",
            "sqeuclidean": "sqeuclidean"
            }

    def __init__(self, p_cluster_number: int, p_init: str = "kmeans++", p_distance: str = "euclidean", p_tol: float = 0.0):
        if p_cluster_number <= 0:
            raise ValueError(
                "El número de clusters k debe ser mayor que 0."
            )
        else:
            self.cluster_number: int = p_cluster_number

        self.centroids: np.ndarray = None
        self.tol: float = p_tol
        self.init: str = p_init # Pendiente de mezclar. Opciones: "kmeans++", "random"

        if p_distance in KMeans.real_distance_metrics:
            self.distance: str = KMeans.real_distance_metrics[p_distance]
        else:
            raise ValueError(f"{p_distance} cannot be used in this class")

    def fit(self, p_X: np.ndarray, p_threshold: float = 0.0, p_seed: int | None = None) -> None:
        N_instancias, D_dimensiones = p_X.shape
        
        if self.cluster_number > N_instancias:
            raise ValueError(
                "El número de clusters k no puede ser mayor "
                "que el número de muestras."
            )

        X: np.ndarray = None

        if self.distance == "cosine":
            X = p_X / (np.linalg.norm(p_X, axis=1, keepdims=True) + 1e-10)
        else:
            X = p_X.copy()

        # 1. Inicialización aleatoria de centroides (Puedes cambiarlo por K-Means++ luego)
        if self.init == "kmeans++":
            self._kmeans_plus_plus(X, p_seed)
        elif self.init == "random":
            self._random_cluster_selection(X, p_seed)

        stop = False

        while not stop:
            # cdist calcula la distancia de todos los puntos a todos los centroides de golpe
            distancias = cdist(X, self.centroids, metric=self.distance)

            # argmin devuelve el índice del centroide más cercano para cada punto (axis=1)
            etiquetas = np.argmin(distancias, axis=1)
            
            # 1. Calculamos la dispersión (suma de errores) de cada cluster
            dispersion_por_cluster = np.zeros(self.cluster_number)
            for k in range(self.cluster_number):
                puntos_k = X[etiquetas == k]
                if len(puntos_k) > 0:
                    # Sumamos las distancias de los puntos de este cluster a su centroide
                    dispersion_por_cluster[k] = np.sum(distancias[etiquetas == k, k])

            # --- FASE DE ACTUALIZACIÓN ---
            nuevos_centroides = np.zeros((self.cluster_number, D_dimensiones))
            for k in range(self.cluster_number):
                # Filtramos los puntos asignados a este cluster 'k' (indexación booleana súper rápida)
                puntos_del_cluster = X[etiquetas == k]
                
                if len(puntos_del_cluster) > 0:
                    nuevos_centroides[k] = puntos_del_cluster.mean(axis=0)
                else:
                    # ESTRATEGIA SCIKIT-LEARN: Coger el punto con mayor error del dataset

                    # 1. Buscamos el cluster más disperso/caótico
                    cluster_mas_disperso = np.argmax(dispersion_por_cluster)

                    # 2. Localizamos los puntos que pertenecen a ese cluster
                    puntos_robables = X[etiquetas == cluster_mas_disperso]
                    distancias_robables = distancias[etiquetas == cluster_mas_disperso, cluster_mas_disperso]

                    # 3. Le robamos la instancia que más se aleja de su centro
                    indice_instancia_robada = np.argmax(distancias_robables)
                    nuevos_centroides[k] = puntos_robables[indice_instancia_robada]

                    # Ponemos su dispersión a 0 por si hay otro cluster vacío, que robe a otro distinto
                    dispersion_por_cluster[cluster_mas_disperso] = 0

            if self.distance == "cosine":
                nuevos_centroides = nuevos_centroides / (np.linalg.norm(nuevos_centroides, axis=1, keepdims=True) + 1e-10)
                
            # --- COMPROBACIÓN DE CONVERGENCIA ---
            desplazamiento = np.linalg.norm(self.centroids - nuevos_centroides)
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

            X: np.ndarray = None

            if self.distance == "cosine":
                X = p_X / (np.linalg.norm(p_X, axis=1, keepdims=True) + 1e-10)
            else:
                X = p_X.copy()

            distances: np.ndarray = cdist(X, self.centroids, metrics=self.distance)
            result = np.argmin(distances, axis=1)

        return result
    
    def _kmeans_plus_plus(self, p_X: np.ndarray, p_seed: int | None = None) -> None:
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
        rng = np.random.default_rng(p_seed)

        # Número de instancias y dimensiones
        n_muestras, n_dimensiones = p_X.shape

        # Comprobación de los parámetros


        # Matriz donde almacenaremos los centroides
        centroides = np.empty(
            (self.cluster_number, n_dimensiones),
            dtype=p_X.dtype
        )

        # ---------------------------------------------------------
        # 1. Seleccionar el primer centroide aleatoriamente
        # ---------------------------------------------------------

        primer_indice = rng.integers(0, n_muestras)

        centroides[0] = p_X[primer_indice]

        # ---------------------------------------------------------
        # 2. Calcular la distancia de cada punto al centroide
        #    seleccionado
        # ---------------------------------------------------------

        distancias_minimas = cdist(p_X, centroides[0:1], metric=self.distance).flatten()

        # ---------------------------------------------------------
        # 3. Seleccionar el resto de centroides
        # ---------------------------------------------------------

        for i in range(1, self.cluster_number):

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
            centroides[i] = p_X[indice]

            # -----------------------------------------------------
            # 4. Actualizar la distancia mínima de cada punto
            # -----------------------------------------------------

            nuevas_distancias = cdist(p_X, centroides[i:i+1], metric=self.distance).flatten()

            distancias_minimas = np.minimum(
                distancias_minimas,
                nuevas_distancias
            )

        self.centroids = centroides

    def _random_cluster_selection(self, p_X: np.ndarray, p_seed: int | None = None):
        rng = None

        if p_seed == None:    
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(p_seed)

        indices_aleatorios = rng.choice(p_X.shape[0], size=self.cluster_number, replace=False)
        self.centroids = p_X[indices_aleatorios].copy()

if __name__ == "__main__":
    print("Inicio")
    X: np.ndarray = np.random.randint(200, size=(10,8))
    print(X, end="\n\nAhora:")
    a: KMeans = KMeans(5)
    a.fit(X)
