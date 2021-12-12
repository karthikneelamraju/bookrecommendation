#!/usr/bin/env python
# coding: utf-8

# In[19]:

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

st.title('Recommended for you!')

# In[3]:


books = pd.read_csv(r'books.csv')

ratings = pd.read_csv(r'ratings.csv')
name = st.sidebar.text_input(''' Enter your user name''')


# # Data Preposessing

# In[4]:


books.columns


# In[5]:


columns = ['id', 'book_id', 'isbn', 'authors', 'original_publication_year', 'title', 'average_rating',
           'ratings_count', 'small_image_url']

books_new = books[columns]

books_new.head()


# In[6]:


books_new.info()


# In[7]:


books_new.isna().sum()


# In[8]:


books_new = books_new.fillna('NA')
books_new.info()


# In[9]:


ratings.isna().sum()


# In[10]:


books_new.to_csv('books_cleaned.csv')


# # Data Modeling:

# In[12]:


#Spliting the data
from sklearn.model_selection import train_test_split

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

print(f"Shape of train data: {train.shape}")
print(f"Shape of test data: {test.shape}")


# In[13]:


#nunique() will return the total number of unique items

book_id = ratings.book_id.nunique() 

user_id = ratings.user_id.nunique()
print('Total books: ' + str(book_id))
print('Total users: ' + str(user_id))


# # Model Building:

# In[14]:


from tensorflow.keras.layers import Dense, Flatten, Input, Embedding, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[16]:


#Embedding layer for books
books_input = Input(shape=[1])#1st Input Layer
embedding_layer_books = Embedding(book_id + 1,10)(books_input)#Embedding layer
embedding_output_books = Flatten()(embedding_layer_books)#Embedding layer output

#Embedding layer for users
users_input = Input(shape=[1])#1st Input Layer
embedding_layer_users = Embedding(user_id + 1,10)(users_input)#Embedding layer
embedding_output_users = Flatten()(embedding_layer_users)#Embedding layer output


# In[17]:


#Concatination and Dense layer

joining_layer = Concatenate()([embedding_output_books, embedding_output_users])
hidden_layer_1 = Dense(128, activation='relu')(joining_layer)
hidden_layer_1 = Dropout(0.5)(hidden_layer_1)

output_layer = hidden_layer_2 = Dense(1)(hidden_layer_1)

model = tf.keras.Model([books_input, users_input], output_layer)


# In[20]:


#Model compilation

optimizer = Adam(lr=0.001, epsilon = 1e-6, amsgrad=True) #epsilon = decay rate
model.compile(optimizer = optimizer, loss = 'mean_squared_error')#Using mean squared error as loss function

model.summary()


# # Training Model

# In[21]:


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1)

model.fit(
    [train.book_id, train.user_id], train.rating, 
    batch_size=64, 
    epochs=15, 
    verbose=1,
    callbacks = [early_stopping],
    validation_data=([test.book_id, test.user_id], test.rating))


# In[22]:


loss = pd.DataFrame(model.history.history)

loss[['loss', 'val_loss']].plot()


# In[24]:


books.head(3)


# In[29]:


#Defining a function that will recommend top 5 books
def recommend(user_id):
    books = pd.read_csv(r'books_cleaned.csv')
    ratings = pd.read_csv(r'ratings.csv')
  
    book_id = list(ratings.book_id.unique()) #grabbing all the unique books
  
    book_arr = np.array(book_id) #geting all book IDs and storing them in the form of an array
    user_arr = np.array([user_id for i in range(len(book_id))])
    prediction = model.predict([book_arr, user_arr])
  
    prediction = prediction.reshape(-1) #reshape to single dimension
    prediction_ids = np.argsort(-prediction)[0:5]

    recommended_books = pd.DataFrame(books.iloc[prediction_ids], columns = ['book_id', 'isbn', 'authors', 'title', 'average_rating' ])
    print('Top 5 recommended books for you: \n')
    return recommended_books


# In[1]:


ratings.user_id.unique().max()


# In[31]:


#Enter a number between 1 and 53424
recommend(789)


# In[ ]:




