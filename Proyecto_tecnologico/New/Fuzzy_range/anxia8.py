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
    f = run_exp_anxia_sim(i+(24*26),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*27),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*28),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*29),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*30),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*31),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*32),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*33),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='both',groups=[10], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*34),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*35),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*36),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*37),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*38),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*39),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*40),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*41),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='positive',groups=[10], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*42),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*43),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*44),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*45),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*46),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*47),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*48),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*49),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.99,
                            chose =1,add ='negative',groups=[10], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)