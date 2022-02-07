#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


# In[2]:


trips = pd.read_csv('data/trips.csv')


# In[3]:


trips


# In[4]:


trips['modechoice'].value_counts()


# In[5]:


utility = pd.read_csv('data/utilityvars.csv')


# In[6]:


utility


# In[7]:


utility.isnull().sum()


# # Exploratory Data Analysis

# In[8]:


# check for problem features and missing data
for col in list(trips):
    print(col,':',trips[col].unique())


# In[9]:


# check for problem features and missing data
for col in list(utility):
    print(col,':',utility[col].unique())


# In[10]:


utility['walkttime'].plot.hist(bins=10, alpha=0.5)


# In[11]:


utility['age'].plot.hist(bins=7, alpha=0.5)


# In[12]:


utility['age'].value_counts()
# we see that most people are aged 56+, followed by the age group between 25-55
# so we have little representation of the younger age group


# In[13]:


# activityid can be removed because it just identifies each person, not relevant for our prediction


# In[14]:


utility['gender'].value_counts()
# we have around 1.2 million more females than males 


# In[15]:


utility['autosuf'].value_counts()
# most common is unsufficient vehicle count, following sufficient count, and little households that don't have vehicle


# In[16]:


utility['income'].value_counts()
# distribution of income seems relatively like normal distribution
# 3 is the average of $60-100k


# In[17]:


utility['income'].plot.hist(bins=5, alpha=0.5)


# In[18]:


utility['tourpurpose'].value_counts()
# work, escort, discretionary are most common purposes of the tour
# university and work-based is the least common


# In[19]:


utility['tourpurpose'].plot.hist(bins=7, alpha=0.5)


# In[20]:


utility['tourmode'].value_counts()
# driving is the most common tour modes, with park and ride and kiss and ride are least common


# In[21]:


utility['tourmode'].plot.hist(bins=12, alpha=0.5)


# In[22]:


utility['targettripmode'].value_counts()


# In[23]:


utility['targettripmode'].plot.hist(bins=12, alpha=0.5)


# # Data Cleaning

# In[24]:


# drop columns with no significant values (e.g. only 1 value, only unique values)
utility = utility.drop(columns=['activityid', 'tourmode'])
utility


# In[25]:


# define categorical variables
utility['gender'] = np.where(utility['gender'] == True, 'female', 'male')

utility['autosuf'] = np.where(utility['autosuf'] == 0, 'no_vehicles', 
                                np.where(utility['autosuf'] == 1, 'insufficient', 'sufficient'))

utility['tourpurpose'] = np.where(utility['tourpurpose'] == 0, 'work',                                     np.where(utility['tourpurpose'] == 1, 'university',                                     np.where(utility['tourpurpose'] == 2, 'school',                                     np.where(utility['tourpurpose'] == 3, 'maintenance',                                     np.where(utility['tourpurpose'] == 4, 'escort',                                     np.where(utility['tourpurpose'] == 5, 'discretionary',                                     np.where(utility['tourpurpose'] == 6, 'work-based', 'cross-border')))))))


# In[26]:


utility.dtypes


# In[27]:


for col in list(utility):
    print(col,':',utility[col].value_counts())


# In[41]:


# define response, categorical, and numerical variables 
Y = utility['targettripmode'] - 1

X1 = utility.select_dtypes(include=['object','bool'])
X2 = utility.select_dtypes(exclude=['object','bool']).drop(columns='targettripmode')
Y


# In[42]:


# show table with expanded categorical columns
expanded_data = []

for col in X1.columns:
    dummies = pd.get_dummies(X1[col], prefix=col)
    expanded_data.append(dummies)

expanded_data = pd.concat(expanded_data, axis=1)
expanded_data


# In[43]:


# all variables with expanded categorical variables
data_encoded = pd.concat([expanded_data, X2, Y], axis=1)
data_encoded


# In[45]:


# uniform subsample
mode1 = data_encoded[data_encoded['targettripmode']==0].sample(n=1000, random_state=1)
mode2 = data_encoded[data_encoded['targettripmode']==1].sample(n=1000, random_state=1)
mode3 = data_encoded[data_encoded['targettripmode']==2].sample(n=1000, random_state=1)
mode4 = data_encoded[data_encoded['targettripmode']==3].sample(n=1000, random_state=1)
mode5 = data_encoded[data_encoded['targettripmode']==4].sample(n=1000, random_state=1)
mode6 = data_encoded[data_encoded['targettripmode']==5].sample(n=1000, random_state=1)
mode7 = data_encoded[data_encoded['targettripmode']==6].sample(n=1000, random_state=1)
mode8 = data_encoded[data_encoded['targettripmode']==7].sample(n=1000, random_state=1)
mode9 = data_encoded[data_encoded['targettripmode']==8].sample(n=1000, random_state=1)
mode10 = data_encoded[data_encoded['targettripmode']==9].sample(n=1000, random_state=1)
mode11 = data_encoded[data_encoded['targettripmode']==10].sample(n=1000, random_state=1)
mode12 = data_encoded[data_encoded['targettripmode']==11].sample(n=1000, random_state=1)


# In[46]:


combined_subsamples = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)


# In[47]:


combined_subsamples


# In[48]:


# rearrange into data array for training
data_array = combined_subsamples.values

X = data_array[:, 0:(data_array.shape[1]-1)]
Y = data_array[:,(data_array.shape[1]-1)]
print(X)
print(Y)


# # Model Building

# In[49]:


# split into train and test sets 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[50]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[51]:


# build xgboost classifier for multiclassification
xgb_model = XGBClassifier(objective='multi:softprob',eval_metric='auc',num_class=12,use_label_encoder=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)],verbose=0)


# # Evaluation

# In[52]:


xgb_model.evals_result()


# In[53]:


# plot training and teating accuracies 

import matplotlib.pyplot as plt
results = xgb_model.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis,results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost AUC')
plt.show()


# In[54]:


y_pred = xgb_model.predict(X_test) 
y_pred = [round(value) for value in y_pred]


# In[55]:


# Evaluation of the model
metrics.multilabel_confusion_matrix(y_test, y_pred)


# In[56]:


# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[57]:


# sensitivity
sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
sensitivity


# In[58]:


# precision
precision = metrics.precision_score(y_test, y_pred, average = 'macro')
precision


# In[59]:


# F1-score
f1 = (2 * precision * sensitivity) / (precision + sensitivity)
f1


# In[60]:


from yellowbrick.classifier import ROCAUC

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'drive_alone_free', 
                                        1: 'drive_alone_pay', 
                                        2: 'hov2_free',
                                        3: 'hov2_pay',
                                        4: 'hov3_free',
                                        5: 'hov3_pay',
                                        6: 'walk',
                                        7: 'bike',
                                        8: 'walk_to_transit',
                                        9: 'park_ride',
                                        10: 'kiss_ride',
                                        11: 'school_bus'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer


# In[61]:


plot_ROC_curve(xgb_model, X_train, y_train, X_test, y_test)


# # Feature Importance

# In[65]:


from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot


# In[66]:


print(xgb_model.feature_importances_)


# In[67]:


pyplot.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
pyplot.show()


# In[68]:


xgb_model.get_booster().feature_names = list(data_encoded.columns)[:-1]
fig, ax = pyplot.subplots(figsize=(5, 10))
plot_importance(xgb_model.get_booster(), ax=ax)
pyplot.show()


# In[ ]:





# # Validation Pipeline

# In[69]:


trips['modechoice'].value_counts()


# In[80]:


# # One-hot encode the categorical variables (the variables only include 0 and -1 don't need to be encoded)
# new_utilityvars = pd.get_dummies(combined_subsamples, columns=["tourpurpose", "tourmode"], prefix=["tourpurpose", "tourmode"])
# new_utilityvars


# In[ ]:


# # Build XGBoost tree model
# new_utilityvars = pd.merge(new_utilityvars.drop(['targettripmode'], axis=1), new_utilityvars[['targettripmode']], left_index=True, right_index=True, how="outer")
# new_utilityvars


# In[ ]:


# df_array = new_utilityvars.values
# X = df_array[:,0:44]
# Y = df_array[:,44]


# In[ ]:


# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[ ]:


# model = xgb.XGBClassifier()
# model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = 0)


# In[ ]:


# y_pred = [round(value) for value in model.predict(X_test)]
# y_test = [round(value) for value in y_test]


# In[73]:


# multilabel confusion matrix
metrics.multilabel_confusion_matrix(y_test, y_pred)


# In[74]:


# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[75]:


# sensitivity
sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
sensitivity


# In[76]:


# precision
precision = metrics.precision_score(y_test, y_pred, average = 'macro')
precision


# In[77]:


# F1-score
f1 = (2 * precision * sensitivity) / (precision + sensitivity)
f1


# In[78]:


# 13*13 confusion matrix
metrics.confusion_matrix(y_test, y_pred)


# In[79]:


# normalized confusion matrix
# metrics.confusion_matrix(y_test, y_pred)[11] / sum(metrics.confusion_matrix(y_test, y_pred)[11])


# In[ ]:




