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
target=np.array(target)
features=np.array(features)


# In[6]:


import sklearn.datasets


# In[7]:


datasets=sklearn.datasets.base.Bunch(data=data,target=target,feature_names=features)


# In[8]:


features1 = pd.DataFrame(datasets.data, columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque',
         'i_d', 'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding',
         'profile_id'])


# In[9]:


target1=pd.DataFrame(datasets.target,columns=['TARGET'])


# In[10]:


data1 = pd.concat([features1, target1], axis=1)


# In[11]:


data2 = data1.corr('pearson')
abs(data2.loc['TARGET']).sort_values(ascending=False)


# In[12]:


X = data1['TARGET'].values.reshape(-1,1)
Y = data1['stator_tooth'].values.reshape(-1,1)


# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# In[14]:


polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(X, Y, s=10)
plt.plot(X, y_poly_pred, color='m')
plt.show()


# In[15]:


polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(X, Y, s=10)
plt.plot(X, y_poly_pred, color='m')
plt.show()


# In[16]:


polynomial_features= PolynomialFeatures(degree=5)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(X, Y, s=10)
plt.plot(X, y_poly_pred, color='m')
plt.show()


# In[17]:


polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(X, Y, s=10)
plt.plot(X, y_poly_pred, color='m')
plt.show()


# In[ ]:




