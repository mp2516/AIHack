import numpy as np
from tqdm import trange

def distance_sq(a, b):
    """
    It does what it says on the tin
    """
    # ARE YOU HAPPY NOW?
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class KMeans2D:
    class Point:
        def __init__(self, location, cluster):
            self.cluster = cluster
            self.location = location

        def x(self):
            return self.location[0]

        def y(self):
            return self.location[1]

        def __getitem__(self, item):
            return self.location[item]

    def __init__(self, coordinates, num_clusters):
        self.num_clusters = num_clusters
        self.points = []
        self.cluster_centroids = []
        self.minx = min(coordinates[0])
        self.maxx = max(coordinates[0])
        self.miny = min(coordinates[1])
        self.maxy = max(coordinates[1])
        for cluster in range(num_clusters):
            self.cluster_centroids.append(
                (np.random.uniform(self.minx, self.maxx), np.random.uniform(self.miny, self.maxy)))

        for point in coordinates:
            min_dist = np.inf
            cluster_idx = -1
            for cluster in range(num_clusters):
                dist_sq = distance_sq(self.cluster_centroids[cluster], point)
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    cluster_idx = cluster
            self.points.append(self.Point(point, cluster_idx))
        self.relocate_centroids()

    def get_cluster_points(self, cluster):
        return (p for p in self.points if p.cluster == cluster)

    def get_num_points(self, cluster):
        return sum(1 for p in self.points if p.cluster == cluster)

    def relocate_centroids(self):
        for cluster in range(self.num_clusters):
            if self.get_num_points(cluster) == 0:
                continue
            xSum, ySum = 0, 0

            for point in self.get_cluster_points(cluster):
                xSum += point.x()
                ySum += point.y()

            self.cluster_centroids[cluster] = (xSum / self.get_num_points(cluster), ySum / self.get_num_points(cluster))

    def assign_clusters(self):
        for point in self.points:
            minDistance, indx = np.inf, -1
            for cluster in range(self.num_clusters):
                dist_sq = distance_sq(point, self.cluster_centroids[cluster])
                if dist_sq < minDistance:
                    minDistance = dist_sq
                    indx = cluster

            point.cluster = indx

    def iterate(self, iterations=1):
        for i in trange(iterations):
            self.relocate_centroids()
            self.assign_clusters()

    def get_x_cluster(self, cluster):
        return [p.x() for p in self.points if p.cluster == cluster]

    def get_y_cluster(self, cluster):
        return [p.y() for p in self.points if p.cluster == cluster]
