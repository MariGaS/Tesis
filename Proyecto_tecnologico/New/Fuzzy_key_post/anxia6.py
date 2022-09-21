from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from vec_function import run_exp_anxia_sim


tr_label = []
for i in range(152):
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

num_test = len(test_labels_anxia)
num_train= 152


arg1 = [1000,1000, 1000,1500,1500,2000,1000,1000, 1000,1500,1500,2000 ] #score1 
arg2 = [1500,1200,800,1000,1200,2000,1500,1200,800,1000,1200,2000] #score2
arg3 = [0.99]*12 #tolerancia 
arg5 = [True, True, True,True,True,True,False, False, False, False, False, False] #fuzzy
arg6 = [True]*12#remove
arg7 = [False]*12#remove
print('Begins experiments')
#arg7 = [0.0003,0.0003,0.0003,0.0001,0.0001,0.0001,0.00005,0.00005,0.00005,0.00001,0.00001,0.00001,0.00003, 0.00003,0.00003,0.007,0.007,0.007,0.0000009,0.000005 ] #score1 
#arg8 = [0.0003,0.0007,0.0001,0.0001,0.0005,0.00005,0.00005,.00001,0.00008,0.00001,0.00006,0.000005,0.00003,0.00008,0.00001,0.007,0.003,0.009,0.0000004,0.000005] #score2
#arg12 = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True]
#arg16 = [True]*20 #dif

# En este no importa si hay en común
for i in range(12):
    f = run_exp_anxia_sim(i+241,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=arg6[i], compress=False, dic =4)

    
for i in range(12):
    f = run_exp_anxia_sim(i+253, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=arg7[i],compress=False, dic =4)
    
#en este si importa 
for i in range(12):
    f = run_exp_anxia_sim(i+265, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = False, fuzzy= arg5[i],remove_stop=arg6[i], compress=False, dic= 4)
    


for i in range(12):
    f= run_exp_anxia_sim(i+277, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = False, fuzzy= arg5[i],remove_stop=arg7[i],compress=False, dic = 4)