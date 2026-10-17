import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist
from metrics import DistanceMetric, CosineMetric, EuclideanMetric

class KMeans:
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

        if p_oversampling_factor is not None and p_oversampling_factor <= 0:
            raise ValueError(
                "El factor de sobremuestreo debe ser positivo."
            )

        if p_rounds is not None and p_rounds <= 0:
            raise ValueError(
                "El número de rondas debe ser positivo."
            )

        self.cluster_number: int = p_cluster_number
        self.centroids: np.ndarray | None = None
        self.tol: float = p_tol
        self.init: str = p_init
        self.metric: DistanceMetric = p_metric

        self.oversampling_factor = p_oversampling_factor
        self.rounds = p_rounds

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

        distancias_minimas = self.metric.get_distance(p_X, centroides[0:1]).flatten()

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

            nuevas_distancias = self.metric.get_distance(p_X, centroides[i:i+1]).flatten()

            distancias_minimas = np.minimum(
                distancias_minimas,
                nuevas_distancias
            )

        self.centroids = centroides
    def _kmeans_parallel(self, p_X: np.ndarray, p_seed: int | None = None) -> None:
        """
        Inicialización de centroides mediante K-Means||.

        1. Selecciona un primer candidato aleatorio.
        2. Realiza varias rondas de sobremuestreo D².
        3. Calcula los pesos de los candidatos.
        4. Reduce los candidatos a k mediante K-Means++
        ponderado.

        Parámetros
        ----------
        p_X : np.ndarray
            Matriz de datos (n_muestras, n_dimensiones).

        p_seed : int | None
            Semilla para reproducibilidad.
        """

        # Generador aleatorio
        rng = np.random.default_rng(p_seed)

        # Dimensiones del conjunto de datos
        n_muestras, n_dimensiones = p_X.shape

        # Factor de sobremuestreo
        l = (
            self.oversampling_factor
            if self.oversampling_factor is not None
            else 2 * self.cluster_number
        )

        # Número de rondas
        rondas = (
            self.rounds
            if self.rounds is not None
            else max(1, int(np.ceil(np.log(n_muestras))))
        )

        # --------------------------------------------------
        # 1. SELECCIONAR EL PRIMER CANDIDATO
        # --------------------------------------------------

        primer_indice = rng.integers(0, n_muestras)

        indices_candidatos = [int(primer_indice)]

        seleccionados = np.zeros(n_muestras, dtype=bool)
        seleccionados[primer_indice] = True

        # Distancias euclídeas al cuadrado
        diferencias = p_X - p_X[primer_indice]

        distancias_minimas = np.sum(
            diferencias ** 2,
            axis=1
        )

        # --------------------------------------------------
        # 2. RONDAS DE SOBREMUESTREO
        # --------------------------------------------------

        for _ in range(rondas):

            # Coste total de los candidatos actuales
            coste_total = np.sum(distancias_minimas)

            if coste_total <= 0:
                break

            # Probabilidad de seleccionar cada punto
            probabilidades = np.minimum(
                1.0,
                l * distancias_minimas / coste_total
            )

            # Evitar seleccionar índices repetidos
            probabilidades[seleccionados] = 0.0

            # Selección independiente de candidatos
            nuevos_indices = np.flatnonzero(
                rng.random(n_muestras) < probabilidades
            )

            if len(nuevos_indices) == 0:
                continue

            indices_candidatos.extend(
                nuevos_indices.tolist()
            )

            seleccionados[nuevos_indices] = True

            # Actualizar la distancia mínima a los candidatos
            nuevos_candidatos = p_X[nuevos_indices]

            nuevas_distancias = cdist(
                p_X,
                nuevos_candidatos,
                metric="sqeuclidean"
            )

            distancias_minimas = np.minimum(
                distancias_minimas,
                np.min(nuevas_distancias, axis=1)
            )

        # --------------------------------------------------
        # 3. OBTENER LOS CANDIDATOS
        # --------------------------------------------------

        candidatos = p_X[indices_candidatos]

        # Si no tenemos suficientes candidatos, añadimos
        # puntos distintos para completar al menos k
        if len(candidatos) < self.cluster_number:

            faltantes = (
                self.cluster_number - len(candidatos)
            )

            disponibles = np.flatnonzero(~seleccionados)

            adicionales = rng.choice(
                disponibles,
                size=faltantes,
                replace=False
            )

            indices_candidatos.extend(
                adicionales.tolist()
            )

            candidatos = p_X[indices_candidatos]

        # --------------------------------------------------
        # 4. CALCULAR LOS PESOS
        # --------------------------------------------------

        # Distancia de cada muestra a cada candidato
        distancias = cdist(
            p_X,
            candidatos,
            metric="sqeuclidean"
        )

        # Candidato más cercano a cada muestra
        asignaciones = np.argmin(
            distancias,
            axis=1
        )

        # Peso = número de muestras asignadas
        pesos = np.bincount(
            asignaciones,
            minlength=len(candidatos)
        ).astype(float)

        # --------------------------------------------------
        # 5. REDUCIR LOS CANDIDATOS A k CENTROIDES
        # --------------------------------------------------

        centroides = self._weighted_kmeans_plus_plus(
            candidatos,
            pesos,
            rng
        )

        self.centroids = centroides

    def _weighted_kmeans_plus_plus(self, p_candidatos: np.ndarray, p_pesos: np.ndarray, p_rng: np.random.Generator) -> np.ndarray:
        """
        Selecciona k centroides mediante K-Means++
        ponderado por los pesos de los candidatos.
        """

        n_candidatos, n_dimensiones = p_candidatos.shape

        centroides = np.empty(
            (self.cluster_number, n_dimensiones),
            dtype=float
        )

        seleccionados = np.zeros(
            n_candidatos,
            dtype=bool
        )

        # --------------------------------------------------
        # 1. PRIMER CENTROIDE
        # --------------------------------------------------

        probabilidades = p_pesos / np.sum(p_pesos)

        primer_indice = p_rng.choice(
            n_candidatos,
            p=probabilidades
        )

        centroides[0] = p_candidatos[primer_indice]
        seleccionados[primer_indice] = True

        # Distancias euclídeas al cuadrado
        distancias_minimas = cdist(
            p_candidatos,
            centroides[0:1],
            metric="sqeuclidean"
        ).flatten()

        # --------------------------------------------------
        # 2. SELECCIONAR LOS RESTANTES CENTROIDES
        # --------------------------------------------------

        for i in range(1, self.cluster_number):

            # Coste ponderado
            costes = p_pesos * distancias_minimas

            costes[seleccionados] = 0.0

            coste_total = np.sum(costes)

            if coste_total <= 0:

                # Si todos los costes son cero,
                # seleccionar un candidato no utilizado
                disponibles = np.flatnonzero(
                    ~seleccionados
                )

                indice = p_rng.choice(disponibles)

            else:

                probabilidades = costes / coste_total

                indice = p_rng.choice(
                    n_candidatos,
                    p=probabilidades
                )

            centroides[i] = p_candidatos[indice]
            seleccionados[indice] = True

            # Actualizar distancias mínimas
            nuevas_distancias = cdist(
                p_candidatos,
                centroides[i:i+1],
                metric="sqeuclidean"
            ).flatten()

            distancias_minimas = np.minimum(
                distancias_minimas,
                nuevas_distancias
            )

        return centroides

    def _random_cluster_selection(self, p_X: np.ndarray, p_seed: int | None = None):
        rng = None

        if p_seed == None:    
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(p_seed)

        indices_aleatorios = rng.choice(p_X.shape[0], size=self.cluster_number, replace=False)
        self.centroids = p_X[indices_aleatorios].copy()

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
