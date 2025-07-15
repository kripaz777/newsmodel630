import streamlit as st
import pandas as pd
import pickle as pk

file = open('LogisticRegression.pickle', 'rb')    
model = pk.load(file)

st.title('News prediction system')

news = st.text_area("Enter news = ")

if st.button('Submit'):
    df = pd.DataFrame(
        {'text':[news]}
    )
    result = model.predict(df['text'])
    st.write(result)