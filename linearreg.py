#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('C:/Users/Vishnu/Desktop/data2.csv',header = None)


# In[3]:


target=pd.read_csv('C:/Users/Vishnu/Desktop/target1.csv',header = None)


# In[4]:


features=pd.read_csv('C:/Users/Vishnu/Desktop/headers.csv',header = None)


# In[5]:


data=np.array(dataset)


# In[6]:


target=np.array(target)


# In[7]:


features=np.array(features)


# In[8]:


import sklearn.datasets


# In[9]:


datasets=sklearn.datasets.base.Bunch(data=data,target=target,feature_names=features)


# In[27]:


features1 = pd.DataFrame(datasets.data, columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque',
         'i_d', 'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding',
         'profile_id'])


# In[28]:


target1=pd.DataFrame(datasets.target,columns=['TARGET'])


# In[29]:


data1 = pd.concat([features1, target1], axis=1)


# In[32]:


data2 = data1.corr('pearson')
abs(data2.loc['TARGET']).sort_values(ascending=False)


# In[14]:


X = data1['TARGET'].values.reshape(-1,1)
Y = data1['stator_tooth'].values.reshape(-1,1)


# In[15]:


X = np.array((X - X.min())/(X.max() - X.min()))
Y = np.array((Y - Y.min())/(Y.max() - Y.min()))


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[18]:


plt.plot(x_train, y_train, 'r.')


# In[19]:


plt.plot(x_test, y_test, 'r.')


# In[20]:


from sklearn.linear_model import LinearRegression


# In[22]:


from sklearn.metrics import mean_squared_error, r2_score


# In[25]:


model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print(rmse)
print(r2)

plt.scatter(X, Y, s=10)
plt.plot(X, y_pred, color='r')
plt.show()


# In[15]:


X = data1['TARGET'].values.reshape(-1,1)
Y = data1['torque'].values.reshape(-1,1)


# In[16]:


X = np.array((X - X.min())/(X.max() - X.min()))
Y = np.array((Y - Y.min())/(Y.max() - Y.min()))


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[19]:


plt.plot(x_train, y_train, 'r.')


# In[20]:


plt.plot(x_test, y_test, 'r.')


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[22]:


model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print(rmse)
print(r2)

plt.scatter(X, Y, s=10)
plt.plot(X, y_pred, color='r')
plt.show()


# In[ ]:




