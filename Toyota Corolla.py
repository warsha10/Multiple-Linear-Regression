#!/usr/bin/env python
# coding: utf-8

# ### MULTIPLE LINEAR REGRESSION 
# Consider only the below columns and prepare a prediction model for predicting Price.
# 
# Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
# 

# In[1]:


# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[6]:


toyota_data=pd.read_csv('ToyotaCorolla.csv',encoding='latin1')
toyota_data


# In[9]:


toyota_data=toyota_data[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight" ]]
toyota_data


# In[15]:


toyota_data=toyota_data.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
toyota_data


# ### INITIAL ANALYSIS

# In[10]:


toyota_data.shape


# In[12]:


#checking for null values 

toyota_data.isna().sum()


# In[13]:


toyota_data.dtypes


# In[ ]:


toyo3=toyo2.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
toyo3


# ### ASSUMPTION CHECK

# #### 1. LINEARITY CHECK

# In[14]:


sns.pairplot(toyota_data)
plt.show()


# #### NO MULTICOLINEARITY

# In[17]:


corr_matrix = toyota_data.corr().round(2)
corr_matrix


# In[18]:


sns.heatmap(data = corr_matrix,annot=True)
plt.show()


# ### Model Building | Training | Evaluating using Statsmodels
# 

# In[19]:


import statsmodels.formula.api as smf
model_1 = smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data = toyota_data).fit()


# In[20]:


# Finding Coefficient parameters
model_1.params


# In[23]:


# Finding tvalues and pvalues
model_1.pvalues


# In[24]:


model_1.rsquared,model_1.rsquared_adj #The contribution of the total input features for the prediction


# In[25]:


model_1 = smf.ols('Price~Age',data = toyota_data).fit()
print('R2 score          : ',round(model_1.rsquared,4))
print('Adjusted R2 score : ',round(model_1.rsquared_adj,4))
print('AIC value         : ',round(model_1.aic,4)) #is an estimator of prediction error 
print('BIC value         : ',round(model_1.bic,4)) ##is an estimator of prediction error 


# In[26]:


model_1 = smf.ols('Price~Age+KM',data = toyota_data).fit()
print('R2 score          : ',round(model_1.rsquared,4))
print('Adjusted R2 score : ',round(model_1.rsquared_adj,4))
print('AIC value         : ',round(model_1.aic,4)) #is an estimator of prediction error 
print('BIC value         : ',round(model_1.bic,4)) ##is an estimator of prediction error 


# In[27]:


model_1 = smf.ols('Price~Age+KM+HP',data = toyota_data).fit()
print('R2 score          : ',round(model_1.rsquared,4))
print('Adjusted R2 score : ',round(model_1.rsquared_adj,4))
print('AIC value         : ',round(model_1.aic,4)) #is an estimator of prediction error 
print('BIC value         : ',round(model_1.bic,4)) ##is an estimator of prediction error 


# In[28]:


model_1 = smf.ols('Price~Age+KM+HP+CC+Doors',data = toyota_data).fit()
print('R2 score          : ',round(model_1.rsquared,4))
print('Adjusted R2 score : ',round(model_1.rsquared_adj,4))
print('AIC value         : ',round(model_1.aic,4)) #is an estimator of prediction error 
print('BIC value         : ',round(model_1.bic,4)) ##is an estimator of prediction error 


# In[29]:


model_1 = smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data = toyota_data).fit()
print('R2 score          : ',round(model_1.rsquared,4))
print('Adjusted R2 score : ',round(model_1.rsquared_adj,4))
print('AIC value         : ',round(model_1.aic,4)) #is an estimator of prediction error 
print('BIC value         : ',round(model_1.bic,4)) ##is an estimator of prediction error 


# In[31]:


model_1.summary()


# ### MODEL BUILDING

# In[32]:


toyota_data.head()


# In[ ]:





# In[37]:


X = toyota_data.drop(labels='Price',axis=1)


# In[38]:


X


# In[41]:


y=toyota_data[['Price']]


# In[42]:


y


# ### MODEL TRAINING

# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


linear_model = LinearRegression() #Initialization
linear_model.fit(X,y) #Model Training


# ### MODEL TESTING

# #### Training data

# In[46]:


y_pred = linear_model.predict(X)
y_pred


# In[47]:


error = y - y_pred
error


# In[48]:


from sklearn.metrics import mean_squared_error


# In[49]:


mean_squared_error(y,y_pred)


# ### MODEL VALIDATION TECHNIQUES

# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=12)


# In[51]:


#Training data
X_train.shape,y_train.shape


# In[52]:


#Test data
X_test.shape,y_test.shape


# In[ ]:





# #### MODEL TRAINING

# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


linear_model_2 = LinearRegression() #Initialization
linear_model_2.fit(X_train,y_train) #Model Training


# #### MODEL TESTING

# #### Training data

# In[55]:


y_train_pred = linear_model.predict(X_train)
y_train_pred


# In[56]:


mean_squared_error(y_train,y_train_pred)


# #### For Test Data

# In[57]:


y_pred_test = linear_model_2.predict(X_test) #unseen by the model during the training time
y_pred_test


# In[58]:


mean_squared_error(y_test,y_pred_test)


# In[ ]:




