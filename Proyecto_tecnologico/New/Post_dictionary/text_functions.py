import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from more_itertools import locate
import numpy as np
# #####--- FUNCTIONS TO READ TRAINING DATA ----########## #


## Functions to read  post from the users ##
def get_urls(path): 
    # Function that return the name of the subjects in the folder 

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles.sort()

    return onlyfiles

def get_text_post_train(user_path):
    tree = ET.parse(user_path)
    root = tree.getroot()
    # list with entries from the user 
    all_post = []
    #iterate recursively over all the sub-tree 
    #source : https://docs.python.org/3/library/xml.etree.elementtree.html
    for post in root.iter('WRITING'): 

        for t in post.iter('TITLE'):
            p = ''
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                if entry[-1] == ' ':
                    p = entry[1:-1] + ' '
                elif entry[-1] == ' ' and entry[-2] == '  ': 
                    p = entry[1:-2] + ' '
                elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                    p = entry[1:-3] + ' '
                else:
                    p = entry[1:] + ' '
        for t in post.iter('TEXT'):
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry != '   ':
                if entry[-1] == ' ' and entry[-2] != ' ' :
                    p = p + entry[1:-1]
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] != ' ': 
                    p = p + entry[1:-2]
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] == ' ': 
                    p = p + entry[1:-3]
                else:
                    p = p + entry[1:]

        all_post.append(p)
    return all_post


def get_text_depression(path): 
    #Function to get all the posts from training set of depression users 

    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents += [document] 


    return all_documents


def get_text_chunk(path): 
    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents += [document] 
    return all_documents
    



def get_dict_position(list1,list2):
    #Function that make a dictionary concatenating all the keywords of the posts
    #list1 is the list with all the keywords of each post 
    #list2 has all the positions of this words
    #merge all the sublist in list1 in one list, the same for list2
    x = [item for sublist in list1 for item in sublist]    
    x = [i[0] for i in x]
    y = [item for sublist in list2 for item in sublist]
    # the list contain the word with the rankings from their respective posts
    position_dic = [[x[i], y[i]] for i in range(0, len(x))]

    return position_dic


def make_final_dic(posi_dict, num_user):
    #posi_dict the dictionary to clean 
    #num_user the number of users to use for this dictionary 
    #contains only words
    words = [posi_dict[i][0] for i in range(len(posi_dict))]
    rankings = [posi_dict[i][1] for i in range(len(posi_dict))]
    #words in a numpy array
    w_array = np.array(words)
    #uniques words in  a numpy array does not preserve original order 
    unique_words = np.unique(w_array)
    #unique_words = set(words)
    
    final_dictionary = []
    #the numpy version of posi_dict
    np_rank = np.array(rankings)
    for i in range(unique_words.shape[0]):
        #we search where this word appears in the posi_dict 
        positions = np.where(w_array == unique_words[i])
        rank_of_word = np_rank[positions[0]].sum()
        final_dictionary.append((unique_words[i],rank_of_word/num_user))
    return final_dictionary   

