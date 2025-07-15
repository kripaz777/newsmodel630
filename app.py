import streamlit as st
import pandas as pd
import pickle as pk
import re,string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()


file = open('LogisticRegression.pickle', 'rb')    
model = pk.load(file)

st.title('News prediction system')

news = st.text_area("Enter news = ")

if st.button('Submit'):
    df = pd.DataFrame(
        {'text':[news]}
    )
    
    df['text'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['text']))
    df['text'] = df['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() ]).lower())
    result = model.predict(df['text'])
    st.write(result)