import numpy as np
from typing import Callable
from metrics import DistanceMetric, EuclideanMetric

InitSignature = Callable[[np.ndarray, int, DistanceMetric, int | None], np.ndarray]

def random_cluster_selection(p_X: np.ndarray, p_cluster_number: int, p_metric: DistanceMetric, p_seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(p_seed)
    return p_X[rng.choice(p_X.shape[0], size=p_cluster_number, replace=False)].copy()

def kmeans_plus_plus(p_X: np.ndarray, p_cluster_number: int, p_metric: DistanceMetric,  p_seed: int | None = None) -> np.ndarray:
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
        (p_cluster_number, n_dimensiones),
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

    distancias_minimas = p_metric.get_distance(p_X, centroides[0:1]).flatten()

    # ---------------------------------------------------------
    # 3. Seleccionar el resto de centroides
    # ---------------------------------------------------------

    for i in range(1, p_cluster_number):

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

        nuevas_distancias = p_metric.get_distance(p_X, centroides[i:i+1]).flatten()

        distancias_minimas = np.minimum(
            distancias_minimas,
            nuevas_distancias
        )

    return centroides

class KMeans_parallel:
    def __init__(self, p_oversampling_factor, p_rounds):
        if p_oversampling_factor is not None and p_oversampling_factor <= 0:
            raise ValueError(
                "El factor de sobremuestreo debe ser positivo."
            )

        if p_rounds is not None and p_rounds <= 0:
            raise ValueError(
                "El número de rondas debe ser positivo."
            )

        self.oversampling_factor = p_oversampling_factor
        self.rounds = p_rounds

    def initialize(self, p_X: np.ndarray, p_cluster_number: int, p_metric: DistanceMetric, p_seed: int | None = None) -> np.ndarray:
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

        if not isinstance(p_metric, EuclideanMetric):
            raise ValueError(
                "K-Means|| clásico requiere EuclideanMetric."
            )

        # Generador aleatorio
        rng = np.random.default_rng(p_seed)

        # Dimensiones del conjunto de datos
        n_muestras, n_dimensiones = p_X.shape

        # Factor de sobremuestreo
        l = (
            self.oversampling_factor
            if self.oversampling_factor is not None
            else 2 * p_cluster_number
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
        if len(candidatos) < p_cluster_number:

            faltantes = (
                p_cluster_number - len(candidatos)
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

        return centroides

    def _weighted_kmeans_plus_plus(self, p_cluster_number: int, p_candidatos: np.ndarray, p_pesos: np.ndarray, p_rng: np.random.Generator) -> np.ndarray:
        """
        Selecciona k centroides mediante K-Means++
        ponderado por los pesos de los candidatos.
        """

        n_candidatos, n_dimensiones = p_candidatos.shape

        centroides = np.empty(
            (p_cluster_number, n_dimensiones),
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

        for i in range(1, p_cluster_number):

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
