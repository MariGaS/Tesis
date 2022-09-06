import pickle

def get_list_key(path):
    #function that gets dictionaries from YAKE using pickle function 
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        fp.close()
    return b

#concatenados version nueva sin shuffle 
#path_1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/keywords/anxia_pos'
#path_2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/keywords/anxia_neg'
# Keywords for depression
#path_3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/keywords/dep_pos'
#path_4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/keywords/dep_neg'

#k5 = get_list_key(path_1)
#k6 = get_list_key(path_2)
#k7 = get_list_key(path_3)
#k8 = get_list_key(path_4)

#concatenado versión nueva shuffle  anorexia 
path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_pos'
path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_neg'
path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_pos2'
path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_neg2'

k1 = get_list_key(path1)
k2 = get_list_key(path2)
k3 = get_list_key(path3)
k4 = get_list_key(path4)

#concatenado versión nueva no-shuffle  anorexia 
path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/anxia_pos'
path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/anxia_neg'
path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/anxia_pos2'
path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/anxia_neg2'

k5 = get_list_key(path5)
k6 = get_list_key(path6)
k7 = get_list_key(path7)
k8 = get_list_key(path8)


print('\nComparate the keywords  anorexia')
for i in range(10):
    print('\nWord pos ', i, 'no-shuffle anorexia ver 1', k5[i])
    print('\nWord neg ', i, 'no-shuffle anorexia ver 1', k6[i])
    print('\nWord pos ', i, 'no-shuffle anorexia ver 2', k7[i])
    print('\nWord neg ', i, 'no-shuffle anorexia ver 2', k8[i])
    print('\nWord pos ', i, 'shuffle anorexia ver 1', k1[i])
    print('\nWord neg ', i, 'shuffle anorexia ver 1', k2[i])
    print('\nWord pos ', i, 'shuffle anorexia ver 2', k3[i])
    print('\nWord neg ', i, 'shuffle anorexia ver 2', k4[i])


#DEPRESSION 
#concatenado versión nueva shuffle  anorexia 
path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_pos'
path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_neg'
path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_pos2'
path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_neg2'

k1 = get_list_key(path1)
k2 = get_list_key(path2)
k3 = get_list_key(path3)
k4 = get_list_key(path4)

#concatenado versión nueva no-shuffle  anorexia 
path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/dep_pos'
path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/dep_neg'
path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/dep_pos2'
path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/dep_neg2'

k5 = get_list_key(path5)
k6 = get_list_key(path6)
k7 = get_list_key(path7)
k8 = get_list_key(path8)


print('\nComparate the keywords  depression')
for i in range(10):
    print('\nWord pos ', i, 'no-shuffle depresssion ver 1', k5[i])
    print('\nWord neg ', i, 'no-shuffle depression ver 1', k6[i])
    print('\nWord pos ', i, 'no-shuffle depression ver 2', k7[i])
    print('\nWord neg ', i, 'no-shuffle depression ver 2', k8[i])
    print('\nWord pos ', i, 'shuffle depression ver 1', k1[i])
    print('\nWord neg ', i, 'shuffle depression ver 1', k2[i])
    print('\nWord pos ', i, 'shuffle depression ver 2', k3[i])
    print('\nWord neg ', i, 'shuffle depression ver 2', k4[i])