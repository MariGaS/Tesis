from User_dictionary.text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from vector_key import run_exp_anxia_sim


### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

pos = 'positive_examples'
neg = 'negative_examples'

all_pos = []
all_neg = []
for i in range(1, 11):
    path_chunk_pos = anxia_train + '/' + pos + '/chunk' + str(i)
    path_chunk_neg = anxia_train + '/' + neg + '/chunk' + str(i)

    temp1 = get_text_chunk(path_chunk_pos)
    temp2 = get_text_chunk(path_chunk_neg)
    if i == 1:
        all_pos = temp1
        all_neg = temp2
    else:
        for j in range(len(temp1)):
            all_pos[j] += temp1[j]

        for j in range(len(temp2)):
            all_neg[j] += temp2[j]

tr_anorexia = [*all_pos, *all_neg]
tr_label = []
for i in range(len(tr_anorexia)):
    if i < 20:
        tr_label.append(1)
    else:
        tr_label.append(0)

test_url_anxia = []
test_labels_anxia = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        test_url_anxia.append(line[:-3])  # only the name of the subject

        test_labels_anxia.append(int(line[-2:-1]))  # only the label
    f.close()

# print(tr_anorexia[0][:10])
# print(test_labels_anxia[:3])

#  ---- TEST-EXTRACTION  --------
test_anxia = []
for i in range(1, 11):

    temp1 = get_text_test_anorexia(anxia_test, test_url_anxia, i)

    if i == 1:
        test_anxia = temp1
        # print("Text extracted from chunk: ", i)
    else:

        for j in range(len(temp1)):
            test_anxia[j] += temp1[j]
        # print("Text extracted from chunk: ", i)

# --------EXPERIMENTS ----------------#

arg1 = [0.007,0.007,0.007,0.007,0.00005,0.00005,0.00005,0.00005,0.0003,0.0003, 0.007,0.007, 0.007, 0.007, 0.00005,0.00005,0.00005,0.00005 ] #score1 
arg2 = [0.003, 0.003, 0.003, 0.003, 0.00001, 0.00001, 0.00001, 0.00001, 0.0001, 0.0001,0.003,0.003,0.003,0.003,0.00001,0.00001,0.00001,0.00001] #score2
arg3 = [0.99]*18 #tolerancia 
arg4 = [False]*18 #dif, it says if the directories has words in common: True means they have, False does not have 
arg5 = [False, True, False,True, False,True,False,True,False,True,True, False, True, False, True, False, True, False] #fuzzy
arg6 = [False, False, True, True,False, False, True, True, True, True, True, True, False, False, False, False, True, True]#remove
arg8 = [1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3]
print('Begins experiments')
# En este no importa si hay en común
for i in range(18):
    f,x,y,z = run_exp_anxia_sim(i+1, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg8[i],dif = arg4[i], fuzzy= arg5[i],remove_stop=arg6[i], concatenate = True)
        
print('End experiments')