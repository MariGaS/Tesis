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


arg1 = [300,320, 250,250,200,200,300,320, 250,250,200,200 ] #score1 
arg2 = [200,200,200,180,180,150,200,200,200,180,180,150] #score2
arg3 = [0.99]*12 #tolerancia 
arg5 = [True, True, True,True,True,True,False, False, False, False, False, False] #fuzzy
arg6 = [True]*12#remove
arg7 = [False]*12#remove
print('Begins experiments')


# En este no importa si hay en común
for i in range(12):
    f = run_exp_anxia_sim(i+49,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = True, fuzzy= arg5[i],remove_stop=arg6[i], compress=False, dic =1)

    
for i in range(12):
    f = run_exp_anxia_sim(i+61, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = True, fuzzy= arg5[i],remove_stop=arg7[i],compress=False, dic =1)
    
#en este si importa 
for i in range(12):
    f = run_exp_anxia_sim(i+73, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=arg6[i], compress=False, dic= 1)
    


for i in range(12):
    f= run_exp_anxia_sim(i+85, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=arg7[i],compress=False, dic = 1)
    