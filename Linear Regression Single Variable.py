#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('homeprices.csv')
df


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[5]:


new_df = df.drop('price',axis='columns')
new_df


# In[6]:


price = df.price
price


# In[7]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)


# ##(1) Predict price of a home with area = 3300 sqr ft

# In[8]:


reg.predict([[3300]])


# In[9]:


reg.coef_


# In[10]:


reg.intercept_


# #Y = m * X + b (m is coefficient and b is intercept)

# In[11]:


3300*135.78767123 + 180616.43835616432


# #(1) Predict price of a home with area = 5000 sqr ft

# In[15]:


reg.predict([[5000]])


# Generate CSV file with list of home price predictions

# In[16]:


area_df = pd.read_csv("areas.csv")
area_df.head(3)


# In[17]:


p = reg.predict(area_df)
p


# In[18]:


area_df['prices']=p
area_df


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area, reg.predict(df[["area"]]), color= 'blue')


# In[ ]:




