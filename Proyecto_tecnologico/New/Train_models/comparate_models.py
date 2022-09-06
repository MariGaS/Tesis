from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import FastText
from gensim.models import utils
from gensim.models import FastText
import gensim
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath

import fasttext
import fasttext.util

model_anxia2 = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/anxiety.model')
print('Load_anorexia')
model_dep2 = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/depresion.model')
print('Load depression')
model_emo2= FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/emotions.model')
print('Load emotions')

model_pre = fasttext.load_model(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/cc.en.300.bin')
print('Load pre_trained')

print('Load_anxia original')
model_anxia = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Model/anxiety.model')
print('Load_dep original')
model_dep = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Model/depresion.model')
model_emo = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Model/emotions.model')

print("The words most similar to emotions from LEXICON")
print('\nFor the news model:')

emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
for i in range(10):
    emotion = emotions[i]
    print('\n The top 20 words more similars to ', emotion, 'in the model_anxia2', 
            model_anxia2.wv.similar_by_word(emotion, topn=20))
    print('\n The top 20 words more similars to ', emotion, 'in the model_anxia1', 
            model_anxia.wv.similar_by_word(emotion, topn=20))
    print('\n The top 20 words more similars to ', emotion, 'in the model_dep2', 
            model_dep2.wv.similar_by_word(emotion, topn=20))
    print('\n The top 20 words more similars to ', emotion, 'in the model_dep1', 
            model_dep.wv.similar_by_word(emotion, topn=20))
    print('\n The top 20 words more similars to ', emotion, 'in the model_emo1', 
            model_emo.wv.similar_by_word(emotion, topn=20))
    print('\n The top 20 words more similars to ', emotion, 'in the model_emo2', 
            model_emo2.wv.similar_by_word(emotion, topn=20))
    print('\n The top 20 words more similars to ', emotion, 'in the pre trained model', 
            model_pre.get_nearest_neighbors(emotion, k=20))
