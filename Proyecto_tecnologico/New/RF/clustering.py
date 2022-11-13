from nltk.cluster import KMeansClusterer
import nltk
import numpy as np

def clustering_features(matrix_emb, num_cluster):
    #convert the matrix_emb to a list of vectors
    list_vectors = list(matrix_emb)

    k_cluster = KMeansClusterer(num_cluster,
                distance = nltk.cluster.util.cosine_distance,
                repeats = 25,
                avoid_empty_clusters = True)
    #return a list with the cluster assign to each vector 
    assigned_clusters = k_cluster.cluster(list_vectors, assign_clusters = True)

    list_group = [] 
    #numpy version of list assigned_clusters
    np_asign = np.array(assigned_clusters)
    for i in range(num_cluster):
        #look where  the are the cluster 
        index = np.where(np_asign == i)
        list_group.append(index[0])

    return list_group

def cluster_add( matrix,num_cluster, matrix_emb):
    list_cluster = clustering_features(matrix_emb,num_cluster)
    for i in range(len(list_cluster)):
        actual_indices = list_cluster[i]
        if i == 0: 
            X_add = np.sum(matrix[:, actual_indices], axis = 1)
        else: 
            X_temp = np.sum(matrix[:,actual_indices], axis = 1)
            X_add = np.column_stack((X_add, X_temp))
            
    return X_add


def clutering_addition(matrix, groups,matrix_emb):
    if len(groups) ==1:
        X_add = cluster_add(matrix, groups[0], matrix_emb)
        matrix = np.column_stack((matrix, X_add))
    else: 
        temp = matrix
        for i in range(len(groups)):
            add_1 = cluster_add(temp, groups[i], matrix_emb)

            matrix = np.column_stack((matrix, add_1))

    return matrix





