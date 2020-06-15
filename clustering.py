import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, X, k, cluster_init_method="random_data", metrics="euclidean", threshold=1000):
        self.X = np.array(X) if X.__class__.__name__!= 'ndarray' else X
        self.k = k
        self.parameters = {
            'cluster_init_method': cluster_init_method,
            'metrics': metrics,
            'threshold': threshold
        }
        self.centroids, self.labels = self.place_centroids()

    def __initialize_centroids(self, bounds):
        if self.parameters['cluster_init_method'] == 'pure_random':
            return np.random.random((self.k,bounds[0].shape[0])) * (np.tile(bounds[1],(self.k,1)) - np.tile(bounds[0],(self.k,1))) + np.tile(bounds[0],(self.k,1))
        elif self.parameters['cluster_init_method'] == 'random_data':
            return self.X[np.random.choice(self.X.shape[0],self.k),:]

    def __assign_labels(self, centroids):
        labels = np.zeros((self.X.shape[0], self.k))
        for cluster in range(self.k):
            labels[:,cluster] = np.sqrt(np.sum((self.X - centroids[cluster,:])**2,axis=1))
        labels=np.argmin(labels, axis=1)
        return labels
    
    def __compute_next_iter(self, centroids):
        next_centroid = np.zeros(centroids.shape)
        labels = self.__assign_labels(centroids)
        for cluster in range(self.k):
            next_centroid[cluster,:] = np.mean(self.X[labels==cluster],axis=0)
        return next_centroid

    def place_centroids(self):
        X_bounds = (np.min(self.X, axis=0), np.max(self.X, axis=0))
        init_centroids = self.__initialize_centroids(X_bounds)
        curr_centroids = self.__compute_next_iter(init_centroids)
        nbr_of_iterations = 0
        while (np.round(curr_centroids,3)!=np.round(init_centroids,3)).all():
            nbr_of_iterations += 1
            curr_centroids = self.__compute_next_iter(curr_centroids)
            if nbr_of_iterations == self.parameters['threshold']:
                break       
        return curr_centroids, labels
    
    def render_data_2d(self):
        for i in range(self.k):
            plt.scatter(self.X[self.labels==i][:,0], self.X[self.labels==i][:,1])
        plt.scatter(self.centroids[:,0], self.centroids[:,1], marker="X", color="red")
        plt.title("2D render")
        plt.xlabel("X")
        plt.ylabel("Y")