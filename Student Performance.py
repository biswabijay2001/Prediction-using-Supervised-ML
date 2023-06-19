#!/usr/bin/env python
# coding: utf-8

# # PREDICTION USING SUPERVISED ML

# IMPORTING REQUIRED LIBRARIES

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,max_error


# READING DATA FROM SYSTEM

# In[2]:


data=pd.read_csv(r"C:\Users\Lenovo\Desktop\Student Dataset.csv")
print("Data imported successfully")
data.head(5)


# Plotting the distribution graph

# In[3]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Student Performance')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# Data Pre-processing

# In[4]:


X = data.iloc[:, :-1].values  
Y = data.iloc[:, 1].values
print(X,Y)


# Spliting train & test data

# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train.shape,Y_train.shape,X_test.shape,Y_test.shape


# Implementing the Linear Regression Model

# In[6]:


regressor=LinearRegression()
regressor.fit(X_train,Y_train)
print("The input data is trained")


# In[7]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# Predicting the output

# In[8]:


print(X_test)
Y_pred = regressor.predict(X_test)
print(Y_pred)


# Compairing the actual vs predicted value

# In[9]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
print(df) 


# In[10]:


hours = [[9.25]]
pred = regressor.predict(hours)
print("Predicted Score ={}".format(pred[0]))


# In[11]:


mse=mean_squared_error(Y_test,Y_pred)
print("Mean squared error: ",mse)
maxe=max_error(Y_test,Y_pred)
print("Max error: ",maxe)
vs=r2_score(Y_test,Y_pred)
print("Variance Score: ",vs) 


# Visualisation of the model

# In[12]:


X_grid=np.arange(1,6,1)
# X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X_grid,Y_test,color='green')
plt.scatter(X_grid,Y_pred,color='pink')
plt.title("Linear Regression")
plt.xlabel("X_grid")
plt.ylabel("Student Performance")
plt.show()


# In[ ]:




