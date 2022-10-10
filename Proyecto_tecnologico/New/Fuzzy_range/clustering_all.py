import numpy as np 
from nltk.cluster import KMeansClusterer
import nltk
from sklearn.metrics import pairwise_distances

#https://stackoverflow.com/questions/57097864/how-to-find-which-text-is-close-to-the-center-of-kmeans-clusters

def clustering_features(matrix_emb, num_cluster):
    #convert the matrix_emb to a list of vectors
    list_vectors = list(matrix_emb)

    k_cluster = KMeansClusterer(num_cluster,
                distance = nltk.cluster.util.cosine_distance,
                repeats = 25,
                avoid_empty_clusters = True)
    #return a list with the cluster assign to each vector 
    assigned_clusters = k_cluster.cluster(list_vectors, assign_clusters = True)
    centroids = k_cluster.means()

    return assigned_clusters, centroids

def find_centroides(matrix_emb, centroids):
    '''
    matrix_emb: matrix of embedings of the vocabulary 
    '''
    centroids = np.array(centroids)
    distances = pairwise_distances(matrix_emb, centroids, 
                               metric='cosine')

    ranking = np.argsort(distances, axis=0)