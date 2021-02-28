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


#funcao para pegar o text no link e limpar
config = Configuration()
config.known_context_patterns = [ArticleContextPattern(attr="class", value="n_text")]


df = pd.read_csv('files/trainFile.csv')
conversion_dict = {0:'Happy',1:'Sad',2:'Neutral'}
df['label'] = df['label'].replace(conversion_dict)


x_train,x_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.25,random_state=7,shuffle=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.75)
vec_train = tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test = tfidf_vectorizer.transform(x_test.values.astype('U'))
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pickle", "wb"))


vectorizer = joblib.load('tfidf_vectorizer.pickle')


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)
pickle.dump(pac, open("pac.pickle", "wb"))

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