from modeloKMeans import *
import numpy as np

def str2Class(p_type: str) -> KMeans | None:
    result: KMeans | None = None
    if p_type == "euclidean":
        result = EuclideanKMeans
    elif p_type == "cosine":
        result = SphericalKMeans

    return result

def str2Initsignature(p_init: str) -> InitSignature:
    result: InitSignature | None = None

    if p_init == "kmeans++":
        result = kmeans_plus_plus
    elif p_init == "random":
        result = random_cluster_selection
    elif p_init == "kameans||":
        result = kmeans_parallel

    return result

def train(p_X: np.ndarray, p_type: str, p_init: str, p_tol: float = 0.0) -> KMeans | None:
    model: KMeans = None

    cluster_number: int = find_best_cluster()

    class_name: KMeans = str2Class(p_type)
    init_name: InitSignature = str2Initsignature(p_init)

    if class_name != None and init_name != None:
        pass

def train_some(p_X: np.ndarray, p_types: np.ndarray, p_init: str, p_tol: float 0.0) -> List[KMeans]:
    models: List[KMeans] = []

    cluster_number: int = find_best_cluster()

    class_names: List[KMeans] = [str2Class(t) for t in p_types if str2Class(t) != None]
    init_name: List[InitSignature] = [str2InitSignature(i) for i in p_init if str2InitSifanture(i) != None]

    pass

def find_cluster_number(p_X: np.ndarray, p_min: int, p_max: int, p_model_name):
    pass
