import numpy as np
from abc import ABC, abstractmethod # Libreria que permite hacer clases abstractas y metodos abstractos
import distance

## INICIALIZACION CON KMEANS++
import numpy as np


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
- Una con KMeans normal
- Otra con KMeans++
- Otra utilizando distancia euclidea
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
        pass

    def fit(self, p_X: np.ndarray) -> None:
        """
        ¿Como realizar la distancia?
        distance.euclidean(v1, v2)
        """
        pass
    
    def predict(self, p_X: np.ndarray) -> np.ndarray:
        pass
        
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
