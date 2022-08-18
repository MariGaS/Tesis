import pickle


def get_list_key(path):

    # function that gets dictionaries from YAKE using pickle function
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        fp.close()
    return b


def get_words_from_kw(kw):

    # only gets the words without the score
    list1 = []
    for i in range(len(kw)):
        list1.append(kw[i][0])
    return list1


def con_all(name_list):
    words = []  # dictionario con palabras
    scores = []  # diccionario con scores

    # diccionario de palabras repetidasd
    repeat_word = []
    # diccionario de cuantas palabras repetidas
    how_many = []

    for i in range(len(name_list)):
        kw = get_list_key(name_list[i])
        if i == 0:
            for j in range(len(kw)):
                word = kw[j][0]
                score = kw[j][1]
                words.append(word)
                scores.append(score)

        else:
            for j in range(len(kw)):
                word = kw[j][0]
                score = kw[j][1]
                if (word in words) == False:
                    # agregamos la palabra
                    words.append(word)
                    # agregamos el score
                    scores.append(score)
                # si se encuentra la palabra
                if word in words:
                    # primero lo agregamos a nuestro diccionario de palabras que se repiten
                    if (word in repeat_word) == False:
                        repeat_word.append(word)
                        how_many.append(1)
                    # buscamos donde se encuentra la palabra en nuestro dictionario limpio
                    # sumamos el score
                    indice = words.index(word)
                    scores[indice] = scores[indice] + score

                    # actualizamos el número de repeticiones
                    # vemos el índice de la palabra
                    if word in repeat_word:
                        ind = repeat_word.index(word)
                        # llevamos el total de veces que aparece el keyword en los diccionarios
                        # si es mayor que 2 si lo tomamos en cuenta

                        how_many[ind] += 1

    # actualizamos el score
    for i in range(len(repeat_word)):
        target_word = repeat_word[i]
        # indice en el diccionario final
        indice = words.index(target_word)

        scores[indice] = scores[indice] / how_many[i]
    #directorio final
    final_dic = []
    for i in range(len(words)):
        # lo agregamos como tupla
        final_dic.append((words[i], scores[i]))

    return final_dic, repeat_word, how_many, scores


# Positive dictionary
name_list = []

for i in range(20):
    name_list.append('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/pos_anxia/key_pos_anxia' + str(i))

dic_pos, rep_pos, how_pos, scores_pos = con_all(name_list)

dic_path1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_pos_anxia'
rep_path1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/rep_pos_anxia'
how_path1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/how_pos_anxia'
scores_path1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/scores_pos_anxia'

with open(dic_path1, 'wb') as f:
    pickle.dump(dic_pos, f)
    f.close()

with open(rep_path1, 'wb') as f:
    pickle.dump(rep_pos, f)
    f.close()

with open(how_path1, 'wb') as f:
    pickle.dump(how_pos, f)
    f.close()

with open(scores_path1, 'wb') as f:
    pickle.dump(scores_pos, f)
    f.close()

# Negative dictionary

name_list2 = []
for i in range(132):
    name_list2.append('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/neg_anxia/key_neg_anxia' + str(i))

dic_neg, rep_neg, how_neg, scores_neg = con_all(name_list2)

dic_path2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_neg_anxia'
rep_path2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/rep_neg_anxia'
how_path2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/how_neg_anxia'
scores_path2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/scores_neg_anxia'

with open(dic_path2, 'wb') as f:
    pickle.dump(dic_neg, f)
    f.close()

with open(rep_path2, 'wb') as f:
    pickle.dump(rep_neg, f)
    f.close()

with open(how_path2, 'wb') as f:
    pickle.dump(how_neg, f)
    f.close()

with open(scores_path2, 'wb') as f:
    pickle.dump(scores_neg, f)
    f.close()
    


## DEPRESSION 
name_list3 = []
for i in range(214):
    name_list3.append('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/pos_dep/key_pos_dep' + str(i))

dic_pos_dep, rep_pos_dep, how_pos_dep, scores_pos_dep = con_all(name_list3)

dic_path3 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_pos_dep'
rep_path3 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/rep_pos_dep'
how_path3 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/how_pos_dep'
scores_path3 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/scores_pos_dep'

with open(dic_path3, 'wb') as f:
    pickle.dump(dic_pos_dep, f)
    f.close()

with open(rep_path3, 'wb') as f:
    pickle.dump(rep_pos_dep, f)
    f.close()

with open(how_path3, 'wb') as f:
    pickle.dump(how_pos_dep, f)
    f.close()

with open(scores_path3, 'wb') as f:
    pickle.dump(scores_pos_dep, f)
    f.close()

# Negative dictionary

name_list4 = []
for i in range(1493):
    name_list4.append('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/neg_dep/key_neg_dep' + str(i))

dic_neg_dep, rep_neg_dep, how_neg_dep, scores_neg_dep = con_all(name_list4)

dic_path4 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_neg_dep'
rep_path4 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/rep_neg_dep'
how_path4 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/how_neg_dep'
scores_path4 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/scores_neg_dep'

with open(dic_path4, 'wb') as f:
    pickle.dump(dic_neg_dep, f)
    f.close()

with open(rep_path4, 'wb') as f:
    pickle.dump(rep_neg_dep, f)
    f.close()

with open(how_path4, 'wb') as f:
    pickle.dump(how_neg_dep, f)
    f.close()

with open(scores_path4, 'wb') as f:
    pickle.dump(scores_neg_dep, f)
    f.close()
    
print('Finish')
        
