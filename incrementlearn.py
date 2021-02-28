import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from deep_translator import GoogleTranslator
import requests
from goose3 import Goose
from goose3.configuration import Configuration, ArticleContextPattern
import pickle
import joblib
import numpy as np

#funcao para pegar o text no link e limpar
config = Configuration()
config.known_context_patterns = [ArticleContextPattern(attr="class", value="n_text")]

classes = np.arange(100)

df = pd.read_csv('files/newDataset.csv')
conversion_dict = {0:'Real',1:'Fake',2:'Neutral'}
df['label'] = df['label'].replace(conversion_dict)


x_train,x_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.25,random_state=7,shuffle=True)


vectorizer = joblib.load('tfidf_vectorizer.pickle')
vec_train = vectorizer.transform(x_train.values.astype('U'))


# pac = PassiveAggressiveClassifier(max_iter=50)
vectorizerPac = joblib.load('pac.pickle')

#incrementa mais estes dados no seu conhecimento
vectorizerPac.partial_fit(vec_train,y_train,classes=None)

pickle.dump(vectorizerPac, open("pac.pickle", "wb"))
# partial_fit

#volta a salvar  depos de treinar
vectorizerPac = joblib.load('pac.pickle')

def getsentment(newtext):
    vec_newstest = vectorizer.transform([newtext])
    y_pred1 = vectorizerPac.predict(vec_newstest)
    return y_pred1[0]

g = Goose(config)
article = g.extract(url='https://www.cnet.com/news/ces-2021-streaming-software-apps-and-services-trends-preview/')
textget = (article.cleaned_text.replace('\ufffd','').replace("\r","").replace("\n",""))

print("--> 👉 "+getsentment(textget))