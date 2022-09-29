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


arg1 = [1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300,1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300 ] #score1 
arg2 = [1500,1200,800,1000,1200,2000,100,100,200,200,2000,700,1500,1200,800,1000,1200,2000,100,100,200,200,2000,700] #score2
arg3 = [0.95]*24 #tolerancia 
arg5 = [True, True, True,True,True,True,True, True, True,True,True,True,False, False, False, False, False, False,False, False, False, False, False, False] #fuzzy

print('Begins experiments')


# En este no importa si hay en común
for i in range(24):
    f = run_exp_anxia_sim(i+2001,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=True, compress=False, dic =1, tf = True)

    f = run_exp_anxia_sim(i+2025, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=False,compress=False, dic =1, tf = True)
    
    f = run_exp_anxia_sim(i+2049, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=True, compress=False, dic= 1, tf= True)

    f= run_exp_anxia_sim(i+2073, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=False,compress=False, dic = 1, tf = True)
    f = run_exp_anxia_sim(i+2097,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=True, compress=True, dic =1, tf = True)

    f = run_exp_anxia_sim(i+2121, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=False,compress=True, dic =1, tf = True)
    
    f = run_exp_anxia_sim(i+2145, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=True, compress=True, dic= 1, tf = True)

    f= run_exp_anxia_sim(i+2169, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=False,compress=True, dic = 1, tf = True)   
    f = run_exp_anxia_sim(i+2193,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=True, compress=False, dic =2, tf = True)

    f = run_exp_anxia_sim(i+2217, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=False,compress=False, dic =2, tf = True)
    
    f = run_exp_anxia_sim(i+2241, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=True, compress=False, dic= 2, tf = True)

    f= run_exp_anxia_sim(i+2265, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=False,compress=False, dic = 2, tf = True)
    f = run_exp_anxia_sim(i+2289,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=True, compress=True, dic =2, tf = True)

    f = run_exp_anxia_sim(i+2313, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =1,dif = True, fuzzy= arg5[i],remove_stop=False,compress=True, dic =2, tf = True)
    
    f = run_exp_anxia_sim(i+2337, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=True, compress=True, dic= 2, tf = True)

    f= run_exp_anxia_sim(i+2361, test_labels_anxia, tr_label,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=False,compress=True, dic = 2, tf = True)