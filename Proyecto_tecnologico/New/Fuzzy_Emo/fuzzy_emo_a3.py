from vector_fuzzy import run_exp_anxia_sim


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

arg3 = [0.95]*24 #tolerancia 
arg5 = [ True, True, False, False]
arg6 = [True, False, True, False]
print('Begins experiments')


# En este no importa si hay en común
for i in range(4):
    f = run_exp_anxia_sim(i+4*33,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict1',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i], only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*34,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict2',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*35,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict3',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*36,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict4',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*37,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict1',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='SVM')

    f = run_exp_anxia_sim(i+4*38,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict2',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='SVM')

    f = run_exp_anxia_sim(i+4*39,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict3',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i], only_bow=False,  tf = False,classificator='SVM')

    f = run_exp_anxia_sim(i+4*40,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict4',tau=arg3[i],
                            chose =1, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='SVM')
    
    f = run_exp_anxia_sim(i+4*41,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict1',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*42,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict2',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*43,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict3',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*44,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict4',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='SVM')

    f = run_exp_anxia_sim(i+4*45,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict1',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],   only_bow=False,tf = False,classificator='SVM')

    f = run_exp_anxia_sim(i+4*46,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict2',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='SVM')

    f = run_exp_anxia_sim(i+4*47,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict3',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i], only_bow=False,  tf = False,classificator='SVM')

    f = run_exp_anxia_sim(i+4*48,test_labels_anxia, tr_label,num_test,num_train,name_dic='dict4',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='SVM')