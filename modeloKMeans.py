import numpy as np
from abc import ABC, abstractmethod # Libreria que permite hacer clases abstractas y metodos abstractos
import distance
from typing import Tuple

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

class Static_list:
    def __init__(self, p_size: Tuple[int]):
        self.size: Tuple[int] = p_size
        self.array: np.ndarray = np.empty(self.size, dtype=np.int32)
        self.array_length: np.ndarray = np.zeros(self.size[0], dtype=np.int32)

    def empty(self) -> None:
        self.array_length = 0

    def append(self, p_row: int, p_instance: np.array) -> None:
        if p_row < self.size[0]:
            current_row_length: int = self.array_length[p_row]
            if current_row_length < self.size[1]:
                self.array[p_row][current_row_length] = p_instance
                self.array_length[p_row] = current_row_length + 1

    def __repr__(self):
        return str(self.array)

class Clusters(Static_list):
    def get_new_centroids(self) -> np.ndarray:
        result: np.ndarray = np.empty((self.size[0], self.size[2]))
        for current_index, current_cluster in enumerate(self.array):
            current_length: int = self.array_length[current_index]
            acc_instance: np.ndarray = np.zeros(self.size[2], dtype=np.float64)

            for current_instance in current_cluster:
                acc_instance += current_instance

            result[current_index] = acc_instance/current_length

        return result

class Centroids:
    def __init__(self, p_array: np.ndarray):
        self.array: np.ndarray = p_array
        self.array_length: int = len(self.array)

    def get_distance_to_cluster(self, p_instance: np.ndarray, p_index: int) -> np.float64:
        if p_index < self.array_length:
            return distance.euclidean(p_instance, self.array[p_index])

        return -1.0

    def get_distance_to_each_cluster(self, p_instance: np.ndarray, p_row_offset: int = 0) -> np.float64:
        iterated_centroids: np.ndarray = self.array[p_row_offset:]

        for current_centroid in iterated_centroids:
            yield distance.euclidean(p_instance, current_centroid)

    def __getitem__(self, p_index):
        return self.array[p_index]

    def __repr__(self):
        return str(self.array)

class KMeans(IKMeans):
    def __init__(self, p_cluster_number: int):
        self.cluster_number: int = p_cluster_number
        self.centroids: np.ndarray = None

    def fit(self, p_X: np.ndarray, p_threshold: float = 0.0) -> None:
        """
        ¿Como realizar la distancia?
        distance.euclidean(v1, v2)
        """
        p_X_row_length: int = len(p_X)
        p_X_column_length: int = len(p_X.T)

        # Random centroid chooser
        rng = np.random.default_rng()
        centroid_rows: np.ndarray = rng.choice(p_X_row_length, size=self.cluster_number, replace=False)
        centroids: Centroids = Centroids(p_X[centroid_rows])

        print(centroids)

        clusters: Clusters = Clusters((self.cluster_number, p_X_row_length, p_X_column_length))

        stop: bool = False

        while not stop:
            for current_instance in p_X:
                min_distance: np.float64 = centroids.get_distance_to_cluster(current_instance, 0)
                closest_index: int = 0

                distance_generator = centroids.get_distance_to_each_cluster(current_instance, 1)
                current_cluster_index: int = 1

                for current_distance in distance_generator:
                    if current_distance < min_distance:
                        min_distance = current_distance
                        closest_index = current_cluster_index

                    current_cluster_index += 1
                
                clusters.append(closest_index, current_instance)

            new_centroids: Centroids = Centroids(clusters.get_new_centroids())

            if new_centroids.equals(centroids, p_threshold):
                stop = True
            else:
                centroids = new_centroids

        self.centroids = new_centroids
    
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
