#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[12]:


mu, sigma = 50, 10
s = np.random.normal(mu, sigma, 100)
s


# In[11]:


plt.hist(s)


# In[4]:


df = pd.DataFrame(s, columns = ['Data'])
df.head()


# In[5]:


for col in df.columns:
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof = 0)


# In[6]:


df['outlier'] = (abs(df['Data_zscore'])> 3).astype(int)


# In[7]:


df.tail()


# In[8]:


df.loc[df.outlier == 1]


# In[ ]:




