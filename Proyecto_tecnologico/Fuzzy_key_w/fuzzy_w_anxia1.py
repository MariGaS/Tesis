from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from vector_key import run_exp_anxia_sim

import pandas as pd

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

with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
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

m = [True]*12
n = [False]*12
f_score = []
best=[]
arg1 = [0.0003,0.0003,0.0001,0.00005,0.00005,0.007,0.0003,0.0003,0.0001,0.00005,0.00005,0.007  ] #score1 
arg2 = [0.0003,0.0001,0.00005,0.00005,0.00001,0.003,0.0003,0.0001,0.00005,0.00005,0.00001,0.003] #score2
arg3 = [0.99]*12 #tolerancia 
arg4 = [*m,*n] #dif, it says if the directories has words in common: True means they have, False does not have 
arg5 = [True, True, True,True,True,True,False, False, False, False, False, False] #fuzzy
arg6 = [True]*12#remove
arg7 = [False]*12#remove
arg8 = [True] *20 #w
print('Begins experiments')
#arg7 = [0.0003,0.0003,0.0003,0.0001,0.0001,0.0001,0.00005,0.00005,0.00005,0.00001,0.00001,0.00001,0.00003, 0.00003,0.00003,0.007,0.007,0.007,0.0000009,0.000005 ] #score1 
#arg8 = [0.0003,0.0007,0.0001,0.0001,0.0005,0.00005,0.00005,.00001,0.00008,0.00001,0.00006,0.000005,0.00003,0.00008,0.00001,0.007,0.003,0.009,0.0000004,0.000005] #score2
#arg12 = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True]
#arg16 = [True]*20 #dif

# En este no importa si hay en común
for i in range(12):
    f = run_exp_anxia_sim(i+1, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,arg1[i],arg2[i],tau=arg3[i], w = arg8[i],
                            chose =1,dif = m[i], fuzzy= arg5[i],remove_stop=arg6[i],train_data = False, compress=False, concatenate = True)
    f_score.append(f)

    
for i in range(12):
    f = run_exp_anxia_sim(i+13, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,arg1[i],arg2[i],tau=arg3[i], w = arg8[i],
                            chose =1,dif = m[i], fuzzy= arg5[i],remove_stop=arg7[i],train_data = False, compress=False, concatenate = True)
    f_score.append(f)
#en este si importa 
for i in range(12):
    f = run_exp_anxia_sim(i+25, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,arg1[i],arg2[i],tau=arg3[i], w= arg8[i],
                            chose =1,dif = n[i], fuzzy= arg5[i],remove_stop=arg6[i],train_data = False, compress=False, concatenate = True)
    f_score.append(f)


for i in range(12):
    f= run_exp_anxia_sim(i+37, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,arg1[i],arg2[i],tau=arg3[i], w= arg8[i],
                            chose =1,dif = n[i], fuzzy= arg5[i],remove_stop=arg7[i],train_data = False, compress=False, concatenate = True)
    f_score.append(f)



        
print('End experiments')

score1 = arg1*4
score2 = arg2*4
tol = arg3*4
rem = [*arg6, *arg7, *arg6, *arg7]
vec = arg5*4
n_d = [*m, *m, *n,*n]
l = [str(x) for x in range(1,49)]
data = { 'score1':score1, 'score2':score2,  'tol': tol, 'remove_stop':rem,'fuzzy':vec,'no_dist': n_d, 'best':best,'f1': f_score}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Anorexia_w_key/Result_w_anxia1.csv',sep='\t')

