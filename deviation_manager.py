import numpy as np

class Deviation_manager:
    def __init__(self, p_cluster_number: int):
        self.cluster_number: int = p_cluster_number

        self.deviation_per_cluster: np.ndarray = None
        self.X: np.ndarray = None
        self.labels: np.ndarray = None
        self.distances: np.ndarray = None

    def update(self, p_X: np.ndarray, p_labels: np.ndarray, p_distances: np.ndarray) -> None:
        self.deviation_per_cluster = np.zeros(self.cluster_number)
        self.X = p_X
        self.labels = p_labels
        self.distances = p_distances

        # 1. Calculamos la dispersión (suma de errores) de cada cluster
        for k in range(self.cluster_number):
            puntos_k = self.X[self.labels == k]
            if len(puntos_k) > 0:
                # Sumamos las distancias de los puntos de este cluster a su centroide
                self.deviation_per_cluster[k] = np.sum(p_distances[self.labels == k, k])

    def get_most_deviated_instance_in_cluster(self) -> np.ndarray:
        # 1. Buscamos el cluster más disperso/caótico
        cluster_mas_disperso = np.argmax(self.deviation_per_cluster)

        # 2. Localizamos los puntos que pertenecen a ese cluster
        puntos_robables = self.X[self.labels == cluster_mas_disperso]
        distancias_robables = self.distances[self.labels == cluster_mas_disperso, cluster_mas_disperso]

        # 3. Le robamos la instancia que más se aleja de su centro
        indice_instancia_robada = np.argmax(distancias_robables)
        result: np.ndarray = puntos_robables[indice_instancia_robada]

        # Ponemos su dispersión a 0 por si hay otro cluster vacío, que robe a otro distinto
        self.deviation_per_cluster[cluster_mas_disperso] = 0

        return result
