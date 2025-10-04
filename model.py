#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

median_value = dataset['test_score(out of 10)'].median()
dataset['test_score(out of 10)'].fillna(median_value, inplace=True)

word_to_num ={'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
dataset['experience'] = dataset['experience'].map(word_to_num)

mode_val = dataset['experience'].mode()
dataset['experience'].fillna(mode_val, inplace=True)

dataset.isnull().sum()

X=dataset.iloc[:,:3]

y=dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
model = regressor.fit(X, y)

with open('model.pkl','wb') as file:
    pickle.dump(model,file)
print("Model saved as model.pkl")

with open('model.pkl','rb') as file:
    load_model=pickle.load(file)

# Test the loaded model
print(
    load_model.predict([[2,9,6]])
    )

get_ipython().system('pip freeze > requirements.txt')


# In[ ]:




