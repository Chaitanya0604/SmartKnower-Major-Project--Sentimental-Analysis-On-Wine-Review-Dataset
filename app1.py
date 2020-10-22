import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('/content/drive/My Drive/SmartknowerML/Machine Learning/wine first 150k.csv') 
df.rename(columns={'Unnamed: 0':'UniqueID'},inplace=True)
df.dropna(axis=0, inplace=True)
df.drop(['UniqueID', 'country',  'designation', 
       'price', 'province', 'region_1', 'region_2', 
        'variety', 'winery'],axis =1,inplace=True)
def partition(x):
  if x >=92:
    return('Positive')
  elif x>=86 and x<92:
    return('Neutral')
  else:
      return('Negative')

score_upd=df['points']

t=score_upd.map(partition)
df['points']=t
df=df[df['points']!='Neutral']

x = df.iloc[:,0].values #Description[Review Text] column as input
y = df.iloc[:,1].values #Points column as output


st.title("Wine Review Classifier")
st.subheader('TFIFD Vectorizer')
st.write('This project is based on Naive Bayes Classifier')


text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x,y)
message = st.text_area("Please enter your Review Text","Type Here ..")
op = text_model.predict([message])
if st.button("Predict"):
  st.title(op)
