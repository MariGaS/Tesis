import numpy as np 
from nltk.cluster import KMeansClusterer
import nltk
from sklearn.metrics import pairwise_distances

from vec_function import *

#POST LEVEL DICTIONARIES 
#---------------------------ANOREXIA-------------------#
#version 1
post_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver1key30'
post_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver1key30'
#version 2
post_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver2key30'
post_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver2key30'

#-------------------------DEPRESSION--------------------#
#version 1 
post_pos_dep1  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_pos_ver1key30'
post_neg_dep1  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_neg_ver1key30'
#version 2
post_pos_dep2  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_pos_ver2key30'
post_neg_dep2  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_neg_ver1key30'

##                                         USER VERSION OF THE DICTIONARIES OF THE USERS                        #
#---------------------------ANOREXIA-------------------#
#version 1
user_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver3'
user_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver3'
#version 2
user_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver4'
user_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver4'

user_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver5'
user_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver5'
#version 2
user_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver6'
user_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
user_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver3'
user_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver3'
#version 2
user_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver4'
user_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver4'

user_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver5'
user_neg_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver5'
#version 2
user_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver6'
user_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver6'

##                                                   CONCATANTE ALL THE TEXT 
#version 1
con_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver3'
con_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver3'
#version 2
con_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver4'
con_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver4'

con_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver5'
con_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver5'
#version 2
con_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver6'
con_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
con_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver3'
con_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver3'
#version 2
con_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver4'
con_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver4'

con_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver5'
con_neg_dep3= '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver5'
#version 2
con_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver6'
con_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver6'

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

    return np.array(assigned_clusters), np.array(centroids)

kw1 = get_list_key(post_pos_anxia1)
kw2 = get_list_key(post_neg_anxia1)
words_pos_anxia1 = get_words_from_kw(kw1)
words_neg_anxia1 = get_words_from_kw(kw2)

post_anxia_emb_matrix_pos1 = get_dictionary_matrix(words_pos_anxia1,1)
post_anxia_emb_matrix_neg1 = get_dictionary_matrix(words_neg_anxia1,1)

post_anxia_emb_matrix_pos2 = get_dictionary_matrix(words_pos_anxia1,3)
post_anxia_emb_matrix_neg2 = get_dictionary_matrix(words_neg_anxia1,3)
### Uppercase model embedding emotions

post_anxia_emb_matrix_pos3 = get_dictionary_matrix(words_pos_anxia1,4)
post_anxia_emb_matrix_neg3 = get_dictionary_matrix(words_neg_anxia1,4)

### Lowercase model embedding anxia 

kw3 = get_list_key(post_pos_anxia2)
kw4 = get_list_key(post_neg_anxia2)


words_pos_anxia2 = get_words_from_kw(kw3)
words_neg_anxia2 = get_words_from_kw(kw4)

post_anxia_emb_matrix_pos4 = get_dictionary_matrix(words_pos_anxia2,1)
post_anxia_emb_matrix_neg4 = get_dictionary_matrix(words_neg_anxia2,1)

### Lowercase model embedding pre-trained 

post_anxia_emb_matrix_pos5 = get_dictionary_matrix(words_pos_anxia2,3)
post_anxia_emb_matrix_neg5 = get_dictionary_matrix(words_neg_anxia2,3)

### Lowercase model embedding emotions

post_anxia_emb_matrix_pos6 = get_dictionary_matrix(words_pos_anxia2,4)
post_anxia_emb_matrix_neg6 = get_dictionary_matrix(words_neg_anxia2,4)

## Depression 

### Uppercase 

kw5 = get_list_key(post_pos_dep1)
kw6 = get_list_key(post_neg_dep1)
words_pos_dep1 = get_words_from_kw(kw5)
words_neg_dep1 = get_words_from_kw(kw6)

### Model embedding dep

post_dep_emb_matrix_pos1 = get_dictionary_matrix(words_pos_dep1,2)
post_dep_emb_matrix_neg1 = get_dictionary_matrix(words_neg_dep1,2)

### Model embedding pre_trained 

post_dep_emb_matrix_pos2 = get_dictionary_matrix(words_pos_dep1,3)
post_dep_emb_matrix_neg2 = get_dictionary_matrix(words_neg_dep1,3)

### Model embedding emotions 

post_dep_emb_matrix_pos3 = get_dictionary_matrix(words_pos_dep1,4)
post_dep_emb_matrix_neg3 = get_dictionary_matrix(words_neg_dep1,4)

### Lowercase

kw7 = get_list_key(post_pos_dep2)
kw8 = get_list_key(post_neg_dep2)
words_pos_dep2 = get_words_from_kw(kw7)
words_neg_dep2 = get_words_from_kw(kw8)

### Model embedding dep

post_dep_emb_matrix_pos4 = get_dictionary_matrix(words_pos_dep2,2)
post_dep_emb_matrix_neg4 = get_dictionary_matrix(words_neg_dep2,2)

### Model embedding pre_trained

post_dep_emb_matrix_pos5 = get_dictionary_matrix(words_pos_dep2,3)
post_dep_emb_matrix_neg5 = get_dictionary_matrix(words_neg_dep2,3)

### Model embedding emotions

post_dep_emb_matrix_pos6 = get_dictionary_matrix(words_pos_dep2,4)
post_dep_emb_matrix_neg6 = get_dictionary_matrix(words_neg_dep2,4)

# Level user

## Anorexia

 ### Uppercase

kw9 = get_list_key(user_pos_anxia1)
kw10 = get_list_key(user_neg_anxia1)
words_pos_anxia3 = get_words_from_kw(kw9)
words_neg_anxia3 = get_words_from_kw(kw10)

### Embedding model anxia 


user_anxia_emb_matrix_pos1 = get_dictionary_matrix(words_pos_anxia3,1)
user_anxia_emb_matrix_neg1 = get_dictionary_matrix(words_neg_anxia3,1)

### Embedding pretrained

user_anxia_emb_matrix_pos2 = get_dictionary_matrix(words_pos_anxia3,3)
user_anxia_emb_matrix_neg2 = get_dictionary_matrix(words_neg_anxia3,3)

### Uppercase model embedding emotions

user_anxia_emb_matrix_pos3 = get_dictionary_matrix(words_pos_anxia3,4)
user_anxia_emb_matrix_neg3 = get_dictionary_matrix(words_neg_anxia3,4)

### Lowercase

kw11 = get_list_key(user_pos_anxia2)
kw12 = get_list_key(user_neg_anxia2)

words_pos_anxia4 = get_words_from_kw(kw11)
words_neg_anxia4 = get_words_from_kw(kw12)

### Model embedding anxia 

user_anxia_emb_matrix_pos4 = get_dictionary_matrix(words_pos_anxia4,1)
user_anxia_emb_matrix_neg4 = get_dictionary_matrix(words_neg_anxia4,1)

### Lowercase model embedding pre-trained 

user_anxia_emb_matrix_pos5 = get_dictionary_matrix(words_pos_anxia4,3)
user_anxia_emb_matrix_neg5 = get_dictionary_matrix(words_neg_anxia4,3)

### Lowercase model embedding emotions

user_anxia_emb_matrix_pos6 = get_dictionary_matrix(words_pos_anxia4,4)
user_anxia_emb_matrix_neg6 = get_dictionary_matrix(words_neg_anxia4,4)

## Depression 

### Uppercase 

kw13 = get_list_key(user_pos_dep1)
kw14 = get_list_key(user_neg_dep1)
words_pos_dep3 = get_words_from_kw(kw13)
words_neg_dep3 = get_words_from_kw(kw14)

### Model embedding dep

user_dep_emb_matrix_pos1 = get_dictionary_matrix(words_pos_dep3,2)
user_dep_emb_matrix_neg1 = get_dictionary_matrix(words_neg_dep3,2)

### Model embedding pre_trained 

user_dep_emb_matrix_pos2 = get_dictionary_matrix(words_pos_dep3,3)
user_dep_emb_matrix_neg2 = get_dictionary_matrix(words_neg_dep3,3)

### Model embedding emotions 

user_dep_emb_matrix_pos3 = get_dictionary_matrix(words_pos_dep3,4)
user_dep_emb_matrix_neg3 = get_dictionary_matrix(words_neg_dep3,4)

### Lowercase

kw15 = get_list_key(user_pos_dep2)
kw16 = get_list_key(user_neg_dep2)
words_pos_dep4 = get_words_from_kw(kw15)
words_neg_dep4 = get_words_from_kw(kw16)

### Model embedding dep

user_dep_emb_matrix_pos4 = get_dictionary_matrix(words_pos_dep4,2)
user_dep_emb_matrix_neg4 = get_dictionary_matrix(words_neg_dep4,2)

### Model embedding pre_trained

user_dep_emb_matrix_pos5 = get_dictionary_matrix(words_pos_dep4,3)
user_dep_emb_matrix_neg5 = get_dictionary_matrix(words_neg_dep4,3)

### Model embedding emotions

user_dep_emb_matrix_pos6 = get_dictionary_matrix(words_pos_dep4,4)
user_dep_emb_matrix_neg6 = get_dictionary_matrix(words_neg_dep4,4)

# Level concatenation

## Anorexia

 ### Uppercase

kw17 = get_list_key(con_pos_anxia1)
kw18 = get_list_key(con_neg_anxia1)
words_pos_anxia5 = get_words_from_kw(kw17)
words_neg_anxia5 = get_words_from_kw(kw18)

### Embedding model anxia 

con_anxia_emb_matrix_pos1 = get_dictionary_matrix(words_pos_anxia5,1)
con_anxia_emb_matrix_neg1 = get_dictionary_matrix(words_neg_anxia5,1)

### Embedding pretrained

con_anxia_emb_matrix_pos2 = get_dictionary_matrix(words_pos_anxia5,3)
con_anxia_emb_matrix_neg2 = get_dictionary_matrix(words_neg_anxia5,3)

### Uppercase model embedding emotions

con_anxia_emb_matrix_pos3 = get_dictionary_matrix(words_pos_anxia5,4)
con_anxia_emb_matrix_neg3 = get_dictionary_matrix(words_neg_anxia5,4)

### Lowercase

kw19 = get_list_key(con_pos_anxia2)
kw20 = get_list_key(con_neg_anxia2)

words_pos_anxia6 = get_words_from_kw(kw19)
words_neg_anxia6 = get_words_from_kw(kw20)

### Model embedding anxia 

con_anxia_emb_matrix_pos4 = get_dictionary_matrix(words_pos_anxia6,1)
con_anxia_emb_matrix_neg4 = get_dictionary_matrix(words_neg_anxia6,1)

### Lowercase model embedding pre-trained 

con_anxia_emb_matrix_pos5 = get_dictionary_matrix(words_pos_anxia6,3)
con_anxia_emb_matrix_neg5 = get_dictionary_matrix(words_neg_anxia6,3)

### Lowercase model embedding emotions

con_anxia_emb_matrix_pos6 = get_dictionary_matrix(words_pos_anxia6,4)
con_anxia_emb_matrix_neg6 = get_dictionary_matrix(words_neg_anxia6,4)

## Depression 

### Uppercase 

kw21 = get_list_key(con_pos_dep1)
kw22 = get_list_key(con_neg_dep1)
words_pos_dep5 = get_words_from_kw(kw21)
words_neg_dep5 = get_words_from_kw(kw22)

### Model embedding dep

con_dep_emb_matrix_pos1 = get_dictionary_matrix(words_pos_dep5,2)
con_dep_emb_matrix_neg1 = get_dictionary_matrix(words_neg_dep5,2)

### Model embedding pre_trained 

con_dep_emb_matrix_pos2 = get_dictionary_matrix(words_pos_dep5,3)
con_dep_emb_matrix_neg2 = get_dictionary_matrix(words_neg_dep5,3)

### Model embedding emotions 

con_dep_emb_matrix_pos3 = get_dictionary_matrix(words_pos_dep5,4)
con_dep_emb_matrix_neg3 = get_dictionary_matrix(words_neg_dep5,4)

### Lowercase

kw23 = get_list_key(con_pos_dep2)
kw24 = get_list_key(con_neg_dep2)
words_pos_dep6 = get_words_from_kw(kw23)
words_neg_dep6 = get_words_from_kw(kw24)

### Model embedding dep

con_dep_emb_matrix_pos4 = get_dictionary_matrix(words_pos_dep6,2)
con_dep_emb_matrix_neg4 = get_dictionary_matrix(words_neg_dep6,2)

### Model embedding pre_trained

con_dep_emb_matrix_pos5 = get_dictionary_matrix(words_pos_dep6,3)
con_dep_emb_matrix_neg5 = get_dictionary_matrix(words_neg_dep6,3)

### Model embedding emotions

con_dep_emb_matrix_pos6 = get_dictionary_matrix(words_pos_dep6,4)
con_dep_emb_matrix_neg6 = get_dictionary_matrix(words_neg_dep6,4)


def get_represententing_words(word_embedding_matrix, num_clusters, dictionary):
    words_centroid = dict()
    array_clusters, centroids = clustering_features(word_embedding_matrix, num_clusters)
    #calculate the cosine similarity between the centroids and the embedings of the words
    similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(word_embedding_matrix, centroids)
    #array with the maximum by column
    max_similarity = np.amax(similarity_matrix, axis = 0)
    #find the index, n_lcuster is the number of cluster corresponding to the maximum in that cluster 
    index, n_cluster = np.where( similarity_matrix == max_similarity)
    
    for i in range(index.shape[0]):
        words_centroid[n_cluster[i]] = dictionary[index[i]]
    
    return words_centroid

# Get clusters 

## Level post : 5 cluster 

cluster_path_5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/10-Cluster'

# con_anxia1 = get_represententing_words(con_anxia_emb_matrix_pos1, 20, words_pos_anxia5)

# with open(cluster_path_5+'/con_anxia_uppercase', "wb") as fp:
#     pickle.dump(con_anxia1, fp)
#     fp.close()

# con_anxia2 = get_represententing_words(con_anxia_emb_matrix_pos2, 20, words_pos_anxia5)
# with open(cluster_path_5+'/con_pre_uppercase', "wb") as fp:
#     pickle.dump(con_anxia2, fp)
#     fp.close()

# con_anxia3 = get_represententing_words(con_anxia_emb_matrix_pos3, 20, words_pos_anxia5)
# with open(cluster_path_5+'/con_emo_uppercase', "wb") as fp:
#     pickle.dump(con_anxia3, fp)
#     fp.close()

# con_anxia4 = get_represententing_words(con_anxia_emb_matrix_pos4, 20, words_pos_anxia6)
# with open(cluster_path_5+'/con_anxia_lowercase', "wb") as fp:
#     pickle.dump(con_anxia4, fp)
#     fp.close()

# con_anxia5 = get_represententing_words(con_anxia_emb_matrix_pos5, 20, words_pos_anxia6)
# with open(cluster_path_5+'/con_pre_lowercase', "wb") as fp:
#     pickle.dump(con_anxia5, fp)
#     fp.close()

# con_anxia6 = get_represententing_words(con_anxia_emb_matrix_pos6, 20, words_pos_anxia6)
# with open(cluster_path_5+'/con_emo_lowercase', "wb") as fp:
#     pickle.dump(con_anxia6, fp)
#     fp.close()

# con_anxia1 = get_represententing_words(con_anxia_emb_matrix_neg1, 20, words_neg_anxia5)

# with open(cluster_path_5+'/con_neg_anxia_uppercase', "wb") as fp:
#     pickle.dump(con_anxia1, fp)
#     fp.close()

# con_anxia2 = get_represententing_words(con_anxia_emb_matrix_neg2, 20, words_neg_anxia5)
# with open(cluster_path_5+'/con_neg_pre_uppercase', "wb") as fp:
#     pickle.dump(con_anxia2, fp)
#     fp.close()

# con_anxia3 = get_represententing_words(con_anxia_emb_matrix_neg3, 20, words_neg_anxia5)
# with open(cluster_path_5+'/con_neg_emo_uppercase', "wb") as fp:
#     pickle.dump(con_anxia3, fp)
#     fp.close()

# con_anxia4 = get_represententing_words(con_anxia_emb_matrix_neg4, 20, words_neg_anxia6)
# with open(cluster_path_5+'/con_neg_anxia_lowercase', "wb") as fp:
#     pickle.dump(con_anxia4, fp)
#     fp.close()

# con_anxia5 = get_represententing_words(con_anxia_emb_matrix_neg5, 20, words_neg_anxia6)
# with open(cluster_path_5+'/con_neg_pre_lowercase', "wb") as fp:
#     pickle.dump(con_anxia5, fp)
#     fp.close()

# con_anxia6 = get_represententing_words(con_anxia_emb_matrix_neg6, 20, words_neg_anxia6)
# with open(cluster_path_5+'/con_neg_emo_lowercase', "wb") as fp:
#     pickle.dump(con_anxia6, fp)
#     fp.close()


# con_anxia1 = get_represententing_words(con_anxia_emb_matrix_neg1, 10, words_neg_anxia5)

# with open(cluster_path_5+'/con_neg_anxia_uppercase', "wb") as fp:
#     pickle.dump(con_anxia1, fp)
#     fp.close()

# con_anxia2 = get_represententing_words(con_anxia_emb_matrix_neg2, 10, words_neg_anxia5)
# with open(cluster_path_5+'/con_neg_pre_uppercase', "wb") as fp:
#     pickle.dump(con_anxia2, fp)
#     fp.close()

# con_anxia3 = get_represententing_words(con_anxia_emb_matrix_neg3, 10, words_neg_anxia5)
# with open(cluster_path_5+'/con_neg_emo_uppercase', "wb") as fp:
#     pickle.dump(con_anxia3, fp)
#     fp.close()

# con_anxia4 = get_represententing_words(con_anxia_emb_matrix_neg4, 10, words_neg_anxia6)
# with open(cluster_path_5+'/con_neg_anxia_lowercase', "wb") as fp:
#     pickle.dump(con_anxia4, fp)
#     fp.close()

# con_anxia5 = get_represententing_words(con_anxia_emb_matrix_neg5, 10, words_neg_anxia6)
# with open(cluster_path_5+'/con_neg_pre_lowercase', "wb") as fp:
#     pickle.dump(con_anxia5, fp)
#     fp.close()

# con_anxia6 = get_represententing_words(con_anxia_emb_matrix_neg6, 10, words_neg_anxia6)
# with open(cluster_path_5+'/con_neg_emo_lowercase', "wb") as fp:
#     pickle.dump(con_anxia6, fp)
#     fp.close()



# con_dep1 = get_represententing_words(con_dep_emb_matrix_neg1, 10, words_neg_dep5)
# with open(cluster_path_5+'/con_neg_dep_uppercase', "wb") as fp:
#     pickle.dump(con_dep1, fp)
#     fp.close()

# con_dep2 = get_represententing_words(con_dep_emb_matrix_neg2, 10, words_neg_dep5)
# with open(cluster_path_5+'/con_neg_dep_pre_uppercase', "wb") as fp:
#     pickle.dump(con_dep2, fp)
#     fp.close()

# con_dep3 = get_represententing_words(con_dep_emb_matrix_neg3, 10, words_neg_dep5)
# with open(cluster_path_5+'/con_neg_dep_emo_uppercase', "wb") as fp:
#     pickle.dump(con_dep3, fp)
#     fp.close()

# con_dep4 = get_represententing_words(con_dep_emb_matrix_neg4, 10, words_neg_dep6)
# with open(cluster_path_5+'/con_neg_dep_lowercase', "wb") as fp:
#     pickle.dump(con_dep4, fp)
#     fp.close()

# con_dep5 = get_represententing_words(user_dep_emb_matrix_neg5, 10, words_neg_dep6)
# with open(cluster_path_5+'/con_neg_dep_pre_lowercase', "wb") as fp:
#     pickle.dump(con_dep5, fp)
#     fp.close()

# con_dep6 = get_represententing_words(con_dep_emb_matrix_neg6, 10, words_neg_dep6)
# with open(cluster_path_5+'/con_neg_dep_emo_lowercase', "wb") as fp:
#     pickle.dump(con_dep6, fp)
#     fp.close()

# user_dep1 = get_represententing_words(user_dep_emb_matrix_pos1,10, words_pos_dep3)
# with open(cluster_path_5+'/user_dep_uppercase', "wb") as fp:
#     pickle.dump(user_dep1, fp)
#     fp.close()

# user_dep2 = get_represententing_words(user_dep_emb_matrix_pos2, 10, words_pos_dep3)
# with open(cluster_path_5+'/user_dep_pre_uppercase', "wb") as fp:
#     pickle.dump(user_dep2, fp)
#     fp.close()

# user_dep3 = get_represententing_words(user_dep_emb_matrix_pos3, 10, words_pos_dep3)
# with open(cluster_path_5+'/user_dep_emo_uppercase', "wb") as fp:
#     pickle.dump(user_dep3, fp)
#     fp.close()

# user_dep4 = get_represententing_words(user_dep_emb_matrix_pos4, 10, words_pos_dep4)
# with open(cluster_path_5+'/user_dep_lowercase', "wb") as fp:
#     pickle.dump(user_dep4, fp)
#     fp.close()

# user_dep5 = get_represententing_words(user_dep_emb_matrix_pos5, 10, words_pos_dep4)
# with open(cluster_path_5+'/user_dep_pre_lowercase', "wb") as fp:
#     pickle.dump(user_dep5, fp)
#     fp.close()

# user_dep6 = get_represententing_words(user_dep_emb_matrix_pos6, 10, words_pos_dep4)
# with open(cluster_path_5+'/user_dep_emo_lowercase', "wb") as fp:
#     pickle.dump(user_dep6, fp)
#     fp.close()



user_anxia1 = get_represententing_words(user_anxia_emb_matrix_neg1, 10, words_neg_anxia3)

with open(cluster_path_5+'/user_neg_anxia_uppercase', "wb") as fp:
    pickle.dump(user_anxia1, fp)
    fp.close()

user_anxia2 = get_represententing_words(user_anxia_emb_matrix_neg2, 10, words_neg_anxia3)
with open(cluster_path_5+'/user_neg_pre_uppercase', "wb") as fp:
    pickle.dump(user_anxia2, fp)
    fp.close()

user_anxia3 = get_represententing_words(user_anxia_emb_matrix_neg3, 10, words_neg_anxia3)
with open(cluster_path_5+'/user_neg_emo_uppercase', "wb") as fp:
    pickle.dump(user_anxia3, fp)
    fp.close()

user_anxia4 = get_represententing_words(user_anxia_emb_matrix_neg4, 10, words_neg_anxia4)
with open(cluster_path_5+'/user_neg_anxia_lowercase', "wb") as fp:
    pickle.dump(user_anxia4, fp)
    fp.close()

user_anxia5 = get_represententing_words(user_anxia_emb_matrix_neg5, 10, words_neg_anxia4)
with open(cluster_path_5+'/user_neg_pre_lowercase', "wb") as fp:
    pickle.dump(user_anxia5, fp)
    fp.close()

user_anxia6 = get_represententing_words(user_anxia_emb_matrix_neg6, 10, words_neg_anxia4)
with open(cluster_path_5+'/user_neg_emo_lowercase', "wb") as fp:
    pickle.dump(user_anxia6, fp)
    fp.close()