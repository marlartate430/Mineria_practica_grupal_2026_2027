import numpy as np
from abc import ABC, abstractmethod # Libreria que permite hacer clases abstractas y metodos abstractos
import distance

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
