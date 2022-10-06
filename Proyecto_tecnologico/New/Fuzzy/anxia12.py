from multiprocessing.util import ForkAwareThreadLock
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


arg1 = [515, 515, 515,515,32,32,32,32 ] #score1 
arg2 = [5745,5745,5745,5745,82,82,82,82] #score2
arg3 = [0.99]*8
arg4 = [3,3,3,3,1,1,1,1] #chose
arg5 = [False,False,False,False,False,False,False,False] #dif 
arg6 = [True, False, True, False, True, False, True, False] #fuzzy 
arg7 = [True, True, False, False, True, True, False] #remove 
arg8 = [True, True, True, True, False, False, False, False] #compress


print('Begins experiments')


# En este no importa si hay en común
for i in range(8):
    f = run_exp_anxia_sim(i+1,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =1,tf = False)
    f = run_exp_anxia_sim(i+9,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =2,tf = False)
    f = run_exp_anxia_sim(i+17,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =3,tf = False)
    f = run_exp_anxia_sim(i+25,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =4,tf = False)
    f = run_exp_anxia_sim(i+33,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =5,tf = False)
    f = run_exp_anxia_sim(i+41,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =6,tf = False)
    f = run_exp_anxia_sim(i+49,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =7,tf = False)
    f = run_exp_anxia_sim(i+57,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =8,tf = False)
    f = run_exp_anxia_sim(i+65,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =1,tf = True)
    f = run_exp_anxia_sim(i+73,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =2,tf = True)
    f = run_exp_anxia_sim(i+81,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =3,tf = True)
    f = run_exp_anxia_sim(i+89,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =4,tf = True)
    f = run_exp_anxia_sim(i+97,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =5,tf = True)
    f = run_exp_anxia_sim(i+105,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =6,tf = True)
    f = run_exp_anxia_sim(i+113,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =7,tf = True)
    f = run_exp_anxia_sim(i+121,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], compress=arg8[i], dic =8,tf = True)
    



    