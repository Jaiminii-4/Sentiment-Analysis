import streamlit as st
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")
    

# st.title("Sentiment Analysis")

selct=st.sidebar.radio("Navigator",["home","speech","text"])

if selct=="home":
    st.title("Sentiment Analysis")
    st.subheader("by Saiyed Huda")
    st.subheader("by Chaudhary Jaimini")

if selct=="speech": 
    st.title("Sentiment Analysis")
    st.header("Write down the text here")
    text_area=st.text_area("enter here")
    
    submit= st.button("submit")
    if submit:
        with open("text_land.txt","w", encoding="utf-8") as file1:
            file1.write(text_area)
        # st.success("text is submitted")
    
    text=open("text_land.txt", encoding='utf-8').read()
    lc=text.lower()
    ct=lc.translate(str.maketrans('','',string.punctuation))
    token_words=word_tokenize(ct,"english")
    final=[]
    for word in token_words:
        if word not in stopwords.words('english'):
            final.append(word)

    emolist=[]
    with open("emotions.txt","r") as file2:
        for line2 in file2:
            clear_line2= line2.replace('\n','').replace(",","").replace("'","").strip()
            word, emotion= clear_line2.split(":")
            if word in final:
                emolist.append(emotion)
            
    st.markdown("""#### The emotion found in above:""")
    st.write(emolist)
    w=Counter(emolist)
    st.markdown("""#### The emotion counter:""")
    st.write(w)

    def sentiment_analise(sentext):
        score=SentimentIntensityAnalyzer().polarity_scores(sentext)
        return score

    sentiment=sentiment_analise(ct)
    st.markdown(f"#### {sentiment} ")
    negative=sentiment["neg"]
    positive=sentiment["pos"]
    if negative > positive:
        st.markdown("""### the sentiment of above is negative""")
    elif positive > negative:
        st.markdown(""" ### the sentiment of above is positive """)
    else: 
        st.markdown(""" ### the sentiment is neutral""")

    def generate_graph(data):
        fig=px.bar(x=list(data.keys()),y=list(data.values()))
        return fig

    st.markdown("""### For Visualization click below""")
    buttu= st.button("click here")
    
    if buttu:
        fig=generate_graph(w)
        st.plotly_chart(fig)



# text sidebar

if selct=="text":
    df=pd.read_csv("sentiment_analysis.csv")

    df["sentiment"]=df["sentiment"].replace({"positive":1,"neutral":0, "negative":2})
    
    # punctuation
    punc = string.punctuation
    def remove_punctuation(text):
        lst = []
        text = text.lower()
        for word in text:
            if word not in punc:
                lst.append(word)
    
        x = lst[:]
        lst.clear()
        return "".join(x)
        
    df["text"] = df["text"].apply(remove_punctuation)
    
    # stopwords
    stop = stopwords.words("english")
    
    def remove_stopwords(text):
        lst = []
    
        for word in text.split():
            if word not in stop:
                lst.append(word)
    
        x = lst[:
               ]
        lst.clear()
        return " ".join(x)
    
    
    df["text"] = df["text"].apply(remove_stopwords)
    
    # stemming
    ps = PorterStemmer()
    
    def stemming(text):
        words = nltk.word_tokenize(text)
        stemmed_words = [ps.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    df["text"]=df["text"].apply(stemming)
    
    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    
    def lemmatizing(text):
        words = nltk.word_tokenize(text)
        lemma_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemma_words)
    
    df["text"]=df["text"].apply(lemmatizing)
    
    # splitting 
    X = df['text']
    Y = df['sentiment']
    
    X_train , X_test ,y_train , y_test = train_test_split(X , Y , train_size = 0.8 , random_state = 0)
    
    # vectorization
    vc = TfidfVectorizer()
    X_train = vc.fit_transform(X_train)
    X_test = vc.transform(X_test)
    
    # using svc
    model = SVC()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_cls = model.predict(X_test)
    
    accuracy_cls = accuracy_score(y_test, y_pred_cls)
    
    f1_cls = f1_score(y_test, y_pred_cls, average='weighted')
    
    # prediction
    def val_to_category(val):
        category_map = {
           0:'neutral',
            1:'positive',
            2:'negative'
         }
        return category_map.get(val,-1)
    
    def make_predictions_svc(text):
        text = stemming(text)
        text = lemmatizing(text)
        text = vc.transform([text])
        val = model.predict(text)
        val = val_to_category(int(val[0]))
        return val
    
    # make_predictions('I feel sorry, I miss you here in the sea beach')
    
    # using random forest
    clf=RandomForestClassifier(n_estimators=100, criterion='gini')
    
    clf.fit(X_train,y_train)
    clf.predict(X_test)
    
    def val_to_category(val):
        category_map = {
           0:'neutral',
            1:'positive',
            2:'negative'
         }
        return category_map.get(val,-1)
    
    def make_predictions_rf(text):
        text = stemming(text)
        text = lemmatizing(text)
        text = vc.transform([text])
        val = clf.predict(text)
        val = val_to_category(int(val[0]))
        return val
        # print("sentiment is : ",val)
    
    # make_predictions_rf('I feel sorry, I miss you here in the sea beach')
    
    
    st.title("Sentiment Analysis using Random Forest and SVC")
    
    st.header("Write down the text here")
    
    text_area=st.text_area("enter here")
    
    submit= st.button("submit")
    
    result1=""
    result2=""
    if submit:
        result1 = make_predictions_rf(text_area)
        result2 = make_predictions_svc(text_area)
    
    st.markdown(f"the sentiment by rf is {result1}")
    st.markdown(f"the sentiment by svc is {result2}")




