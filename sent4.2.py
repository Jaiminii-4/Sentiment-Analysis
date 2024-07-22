#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("sentiment_analysis.csv")


# In[3]:


df.head()


# In[4]:


#eda


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.describe().T.style.background_gradient(axis=1)


# In[9]:


df_columns=df.columns
for col in df.columns:
    print(col)


# In[10]:


df1= df.loc[:,("text", "sentiment")]


# In[11]:


df1.head()


# ### Visualization

# In[12]:


df.sentiment.value_counts().plot(kind='bar')
plt.show()


# In[13]:


df.Year.value_counts().plot(kind='bar')
plt.show()


# In[14]:


sns.displot(df.Platform ,fill=True)


# In[15]:


sns.histplot(df['Month'],bins=20,color='Blue',edgecolor='Red',kde=True)
plt.xlabel("Year")
plt.ylabel("platform")


# In[16]:


sns.scatterplot(data = df , x ='Year' , y ='Platform')


# In[ ]:





# In[17]:


df_columns=df.columns
for col in df.columns:
    print(col)


# In[18]:


df["sentiment"].value_counts()
df["sentiment"]=df["sentiment"].replace({"positive":1,"neutral":0, "negative":2})


# In[19]:


df.groupby(['Platform','Year']).sentiment.value_counts().sort_values(ascending = False).head(10).plot(kind = 'pie',autopct = '%1.1f%%',ylabel='')
plt.show()


# ## Preparing

# ### Remove punctuation

# In[20]:


import string


# In[21]:


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

df.head()


# ### Removing stop words

# In[22]:


from nltk.corpus import stopwords


# In[23]:


#extra
import nltk
nltk.download('stopwords')


# In[24]:


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


# In[25]:


df.head()


# ### Stemmering the data

# In[26]:


#extra
get_ipython().system('pip install nltk')
nltk.download('punkt')


# In[27]:


from nltk.stem import PorterStemmer
import nltk
ps = PorterStemmer()

def stemming(text):
    words = nltk.word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)

df["text"]=df["text"].apply(stemming)


# In[28]:


df.head()


# ### Lemmatizing the data

# In[29]:


from nltk.stem import WordNetLemmatizer


# In[30]:


nltk.download("wordnet")


# In[31]:


lemmatizer = WordNetLemmatizer()

def lemmatizing(text):
    words = nltk.word_tokenize(text)
    lemma_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemma_words)

df["text"]=df["text"].apply(lemmatizing)


# In[32]:


df["text"]


# In[33]:


df.head()


# ## Splitting the data

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC


# In[35]:


X = df['text']
Y = df['sentiment']


# In[36]:


X_train , X_test ,y_train , y_test = train_test_split(X , Y , train_size = 0.8 , random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[38]:


vc = TfidfVectorizer()
X_train = vc.fit_transform(X_train)
X_test = vc.transform(X_test)


# ## Using SVC model 

# In[39]:


model = SVC()
model.fit(X_train, y_train)

# Make predictions
y_pred_cls = model.predict(X_test)

accuracy_cls = accuracy_score(y_test, y_pred_cls)

f1_cls = f1_score(y_test, y_pred_cls, average='weighted')


# In[40]:


y_pred_cls


# In[41]:


def val_to_category(val):
    category_map = {
       0:'neutral',
        1:'positive',
        2:'negative'
     }
    return category_map.get(val,-1)


# In[42]:


def make_predictions(text):
    text = stemming(text)
    text = lemmatizing(text)
    text = vc.transform([text])
    val = model.predict(text)
    val = val_to_category(int(val[0]))
    print("sentiment is : ",val)


# #### Making prediction

# In[43]:


make_predictions('I feel sorry, I miss you here in the sea beach')


# #### Score

# In[44]:


f1_cls


# In[45]:


accuracy_cls


# In[ ]:





# ## Using Random Forest

# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


clf=RandomForestClassifier(n_estimators=100, criterion='gini')


# In[48]:


clf.fit(X_train,y_train)


# In[49]:


clf.predict(X_test)


# In[50]:


def val_to_category(val):
    category_map = {
       0:'neutral',
        1:'positive',
        2:'negative'
     }
    return category_map.get(val,-1)


# In[51]:


def make_predictions3(text):
    text = stemming(text)
    text = lemmatizing(text)
    text = vc.transform([text])
    val = clf.predict(text)
    val = val_to_category(int(val[0]))
    print("sentiment is : ",val)


# #### Making prediction

# In[55]:


make_predictions3('I feel sorry, I miss you here in the sea beach')


# #### Score

# In[56]:


score3=clf.score(X_test,y_test)


# In[57]:


score3


# In[2]:


get_ipython().system('jupyter nbconvert --to python sent4byja.ipynb')


# In[ ]:




