import numpy as np

def euclidean(p_v1: np.array, p_v2: np.array) -> np.float64:
    """
    Función que calcula la distancia EUCLIDEA de forma MANUAL,
    para que NO realice la RAIZ CUADRADA. Utilizar el producto
    escalar de los dos vectores devuelve ese resultado

    v1 · v1 = v1[0] * v1[0] + v1[1] * v1[1] + ... + v1[N] * v1[N]
    """

    difference_vector: np.array = p_v2 - p_v1
    
    return np.dot(difference_vector, difference_vector)

def cosine_similarity(p_v1: np.array, p_v2: np.array) -> np.float64:
    """
    DUDAS:
    En el preproceso escalarlo sobre el máximo (Respecto al módulo)
    más grande (No es totalmente real). y realizar el producto de los
    elementos del vector.

    Usar la similitud del coseno normal
    """
    pass
