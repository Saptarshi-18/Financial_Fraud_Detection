#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ## Importing Dataset

# In[2]:


df = pd.read_csv("D:\FinancialFraudDetectionModels\dataset.csv",low_memory = False)


# ## Printing first 5 rows of the dataset

# In[3]:


df.head(n=5)


# ## Dropping unnecessary columns

# In[4]:


new_df= df.drop(['step','nameOrig','nameDest','isFlaggedFraud','oldbalanceDest','newbalanceDest'], axis = 1)


# ## Printing the first 10 rows of the new formed dataset

# In[5]:


new_df.head(n = 10)


# ## Converting Categorical data into a numeric data

# In[6]:


new_df['type']=new_df['type'].map({'PAYMENT':1, 'TRANSFER':4, 'CASH_OUT':2, 'DEBIT':5, 'CASH_IN':3})


# ## Printing the new dataset after the necessary changes

# In[7]:


new_df.head()


# ## Extracting Features (X) and Target (y) from DataFrame

# In[8]:


X = new_df.iloc[:,:-1].values
y = new_df.iloc[:,-1].values


# ## Splitting Data into Training and Test Sets 

# In[9]:


X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


# ## Feature Scaling with Standardization for Machine Learning

# In[10]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

