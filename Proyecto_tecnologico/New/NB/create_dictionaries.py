from vec_function import *
import pickle


arg1 = [100,500,100, 500, 1000, 800, 200, 1000, 1500, 300] #numfeature 1
arg2 = [100,200,200, 200, 1000, 800, 100, 1200, 1200, 700] #numfeature 2
arg3 = [1,2,2, 2, 2, 5, 2, 1,2,2]    #dic
arg4 = [False,True,True,False, True, False, True, False, False, False]    #dif

for i in range(10):
    dic1, dic2 = dictionaries_to_txt_anorexia(arg3[i], arg1[i], arg2[i], arg3[i])
    dictionary = dic1+ dic2
    name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/NB/dictionaries_anxia/anxia_' + str(i) + '.txt'
    with open(name_key, 'wb') as f:
        pickle.dump(dictionary, f)
        f.close()
