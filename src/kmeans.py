import numpy as np
from tqdm import trange


def distance_sq(a, b):
    if len(a) != len(b):
        raise ValueError("A and b must have identical dimensions")
    return sum(((a[i] - b[i])**2 for i in range(len(a))))


class KMeans:
    class Point:
        def __init__(self, location, cluster):
            self.cluster = cluster
            self.location = location

        def x(self):
            return self.location[0]

        def y(self):
            return self.location[1]

        def __len__(self):
            return len(self.location)

        def __getitem__(self, item):
            return self.location[item]

    def __init__(self, coordinates, num_clusters,weights):
        self.dims = len(coordinates[0])
        self.num_clusters = num_clusters
        self.points = []
        self.cluster_centroids = []
        self.normalization = []
        for cluster in range(num_clusters):
            cluster_p = []
            for d in range(self.dims):
                cluster_p.append(np.random.uniform(0, 1))
            self.cluster_centroids.append(cluster_p)

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
            sm = [0 for i in range(self.dims)]

            for point in self.get_cluster_points(cluster):
                for i in range(self.dims):
                    sm[i] += point[i]
            d = []
            for i in range(self.dims):
                d.append(sm[i]/self.get_num_points(cluster))
            self.cluster_centroids[cluster] =d

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
            self.assign_clusters()
            self.relocate_centroids()

    def get_x_cluster(self, cluster):
        return [p.x() for p in self.points if p.cluster == cluster]

    def get_y_cluster(self, cluster):
        return [p.y() for p in self.points if p.cluster == cluster]
