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

print('Begins experiments')


# En este no importa si hay en común
for i in range(24):
    f = run_exp_anxia_sim(i+(24*218),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*219),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*220),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*221),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*222),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*223),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*224),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*225),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*226),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*227),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*228),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*229),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*230),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*231),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*232),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*233),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*234),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*235),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*236),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*237),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*238),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*239),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*240),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*241),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)