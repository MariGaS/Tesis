import pickle
from readline import write_history_file
from unicodedata import name 
def get_words_from_kw(kw):
    list1 = []
    for i in range(len(kw)):
        list1.append(kw[i][0])
    return list1


def get_list_key(path):
    # function that gets dictionaries from YAKE using pickle function
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        fp.close()
    return b


def dict_scores(d1, d2, feature1, feature2, no_distintc, con):

    if no_distintc == True:
        # obtain the
        kw1 = d1[:feature1]
        kw2 = d2[:feature2]

        l1 = get_words_from_kw(kw1)
        l2 = get_words_from_kw(kw2)

        return l1, l2

    else:

        dictionary1 = []
        dictionary2 = []
        l1 = get_words_from_kw(d1)
        l2 = get_words_from_kw(d2)
        i = 0
        while len(dictionary1) != feature1:
            w1 = l1[i]
            # revisamos si no está en el diccionario negativo
            if (w1 in l2) == False:
                dictionary1.append(w1)

            else:
                # en donde se encuentra en la lista l2
                indice = l2.index(w1)
                # score en la lista 1
                rel1 = d1[i][1]
                # score en la lista 2
                rel2 = d2[indice][1]
                # si el score de la lista 1 es menor que en la 1 la dejamos en lista 1
                if con == True: 
                    if rel1< rel2:
                        dictionary1.append(w1)
                else: 
                    if rel2 < rel1:
                        dictionary1.append(w1)
            i +=1 
        # dictionario negativo son todas las palabras que no están en dictionariopos
        dictionary2 = [x for x in l2 if x not in dictionary1][:feature2]
    return dictionary1, dictionary2


def write_dictionary(path, dic):
    with open(path, "w") as f:
        for i in range(len(dic)):
            f.write(str(dic[i]) + '\n')
        f.close()


def get_dictionary(name_path1, name_path2, dic1, dic2, score1, score2, no_distintic, con):
    dictionary1, dictionary2 = dict_scores(dic1,dic2,score1, score2, no_distintc=no_distintic, con = con)

    write_dictionary(name_path1, dictionary1)
    write_dictionary(name_path2, dictionary2)

