#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


# In[2]:


trips = pd.read_csv('data/trips.csv')
trips


# In[3]:


utility = pd.read_csv('data/utilityvars.csv')
utility


# ## Exploratory Data Analysis

# In[4]:


trips.dtypes.unique()


# In[5]:


utility.dtypes.unique()


# In[6]:


# check for problem features and missing data
for col in list(trips):
    print(col,':',trips[col].unique())


# In[7]:


# check for problem features and missing data
for col in list(utility):
    print(col,':',utility[col].unique())


# In[8]:


utility['walkttime'].plot.hist(bins=10, alpha=0.5)


# In[9]:


utility['age'].plot.hist(bins=7, alpha=0.5)


# In[10]:


utility['age'].value_counts()
# we see that most people are aged 56+, followed by the age group between 25-55
# so we have little representation of the younger age group


# In[11]:


# activityid can be removed because it just identifies each person, not relevant for our prediction


# In[12]:


utility['gender'].value_counts()
# we have around 1.2 million more females than males 


# In[13]:


utility['autosuf'].value_counts()
# most common is unsufficient vehicle count, following sufficient count, and little households that don't have vehicle


# In[14]:


utility['income'].value_counts()
# distribution of income seems relatively like normal distribution
# 3 is the average of $60-100k


# In[15]:


utility['income'].plot.hist(bins=5, alpha=0.5)


# In[16]:


utility['tourpurpose'].value_counts()
# work, escort, discretionary are most common purposes of the tour
# university and work-based is the least common


# In[17]:


utility['tourpurpose'].plot.hist(bins=7, alpha=0.5)


# In[18]:


utility['tourmode'].value_counts()
# driving is the most common tour modes, with park and ride and kiss and ride are least common


# In[19]:


utility['tourmode'].plot.hist(bins=12, alpha=0.5)


# In[20]:


utility['targettripmode'].value_counts()


# In[21]:


utility['targettripmode'].plot.hist(bins=12, alpha=0.5)


# ## Data Cleaning

# In[4]:


# drop columns with no significant values (e.g. only 1 value, only unique values)
# dropped tourmode due to high correlation
utility = utility.drop(columns=['activityid', 'tourmode'])
utility


# In[5]:


# define categorical variables
utility['gender'] = np.where(utility['gender'] == True, 'female', 'male')

utility['autosuf'] = np.where(utility['autosuf'] == 0, 'no_vehicles', 
                                np.where(utility['autosuf'] == 1, 'insufficient', 'sufficient'))

utility['tourpurpose'] = np.where(utility['tourpurpose'] == 0, 'work',                                     np.where(utility['tourpurpose'] == 1, 'university',                                     np.where(utility['tourpurpose'] == 2, 'school',                                     np.where(utility['tourpurpose'] == 3, 'maintenance',                                     np.where(utility['tourpurpose'] == 4, 'escort',                                     np.where(utility['tourpurpose'] == 5, 'discretionary',                                     np.where(utility['tourpurpose'] == 6, 'work-based', 'cross-border')))))))
                                    
    
# utility['tourmode'] = np.where(utility['tourmode'] == 1, 'drive_alone_free', \
#                                 np.where(utility['tourmode'] == 3, 'hov2',\
#                                 np.where(utility['tourmode'] == 5, 'hov3',\
#                                 np.where(utility['tourmode'] == 7, 'walk',\
#                                 np.where(utility['tourmode'] == 8, 'bike',\
#                                 np.where(utility['tourmode'] == 9, 'walk_to_transit',\
#                                 np.where(utility['tourmode'] == 10, 'park_ride',\
#                                 np.where(utility['tourmode'] == 11, 'kiss_ride','school_bus'))))))))

# utility['targettripmode'] = np.where(utility['targettripmode'] == 1, 'drive_alone_free', \
#                                 np.where(utility['targettripmode'] == 2, 'drive_alone_pay',\
#                                 np.where(utility['targettripmode'] == 3, 'hov2_free',\
#                                 np.where(utility['targettripmode'] == 4, 'hov2_pay',\
#                                 np.where(utility['targettripmode'] == 5, 'hov3_free',\
#                                 np.where(utility['targettripmode'] == 6, 'hov3_pay',\
#                                 np.where(utility['targettripmode'] == 7, 'walk',\
#                                 np.where(utility['targettripmode'] == 8, 'bike',\
#                                 np.where(utility['targettripmode'] == 9, 'walk_to_transit',\
#                                 np.where(utility['targettripmode'] == 10, 'park_ride',\
#                                 np.where(utility['targettripmode'] == 11, 'kiss_ride','school_bus')))))))))))


# In[6]:


utility.dtypes


# In[7]:


for col in list(utility):
    print(col,':',utility[col].value_counts())


# In[8]:


# define response, categorical, and numerical variables 
Y = utility['targettripmode'] - 1

X1 = utility.select_dtypes(include=['object','bool'])
X2 = utility.select_dtypes(exclude=['object','bool']).drop(columns='targettripmode')
Y


# In[9]:


# show table with expanded categorical columns
expanded_data = []

for col in X1.columns:
    dummies = pd.get_dummies(X1[col], prefix=col)
    expanded_data.append(dummies)

expanded_data = pd.concat(expanded_data, axis=1)
expanded_data


# In[10]:


# all variables with expanded categorical variables
data_encoded = pd.concat([expanded_data, X2, Y], axis=1)
data_encoded


# In[107]:


data_encoded['targettripmode'].value_counts()


# In[11]:


# check distribution of full dataset 
props = data_encoded['targettripmode'].value_counts()/len(data_encoded['targettripmode'])
props


# In[160]:


# imbalanced subsample
size = 1000000
mode1 = data_encoded[data_encoded['targettripmode']==0].sample(n=int(round(size*props[0])), random_state=1)
mode2 = data_encoded[data_encoded['targettripmode']==1].sample(n=int(round(size*props[1])), random_state=1)
mode3 = data_encoded[data_encoded['targettripmode']==2].sample(n=int(round(size*props[2])), random_state=1)
mode4 = data_encoded[data_encoded['targettripmode']==3].sample(n=int(round(size*props[3])), random_state=1)
mode5 = data_encoded[data_encoded['targettripmode']==4].sample(n=int(round(size*props[4])), random_state=1)
mode6 = data_encoded[data_encoded['targettripmode']==5].sample(n=int(round(size*props[5])), random_state=1)
mode7 = data_encoded[data_encoded['targettripmode']==6].sample(n=int(round(size*props[6])), random_state=1)
mode8 = data_encoded[data_encoded['targettripmode']==7].sample(n=int(round(size*props[7])), random_state=1)
mode9 = data_encoded[data_encoded['targettripmode']==8].sample(n=int(round(size*props[8])), random_state=1)
mode10 = data_encoded[data_encoded['targettripmode']==9].sample(n=int(round(size*props[9])), random_state=1)
mode11 = data_encoded[data_encoded['targettripmode']==10].sample(n=int(round(size*props[10])), random_state=1)
mode12 = data_encoded[data_encoded['targettripmode']==11].sample(n=int(round(size*props[11])), random_state=1)


# In[66]:


# uniform subsample
# mode1 = data_encoded[data_encoded['targettripmode']==0].sample(n=1000, random_state=1)
# mode2 = data_encoded[data_encoded['targettripmode']==1].sample(n=1000, random_state=1)
# mode3 = data_encoded[data_encoded['targettripmode']==2].sample(n=1000, random_state=1)
# mode4 = data_encoded[data_encoded['targettripmode']==3].sample(n=1000, random_state=1)
# mode5 = data_encoded[data_encoded['targettripmode']==4].sample(n=1000, random_state=1)
# mode6 = data_encoded[data_encoded['targettripmode']==5].sample(n=1000, random_state=1)
# mode7 = data_encoded[data_encoded['targettripmode']==6].sample(n=1000, random_state=1)
# mode8 = data_encoded[data_encoded['targettripmode']==7].sample(n=1000, random_state=1)
# mode9 = data_encoded[data_encoded['targettripmode']==8].sample(n=1000, random_state=1)
# mode10 = data_encoded[data_encoded['targettripmode']==9].sample(n=1000, random_state=1)
# mode11 = data_encoded[data_encoded['targettripmode']==10].sample(n=1000, random_state=1)
# mode12 = data_encoded[data_encoded['targettripmode']==11].sample(n=1000, random_state=1)


# In[67]:


combined_subsamples = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)

combined_subsamples


# In[68]:


# check sample distribution is the same as full dataset
samp_props = combined_subsamples['targettripmode'].value_counts()/len(combined_subsamples['targettripmode'])
samp_props


# In[69]:


# rearrange into data array for training

data_array = combined_subsamples.values

X = data_array[:, 0:(data_array.shape[1]-1)]
Y = data_array[:,(data_array.shape[1]-1)]
print(X)
print(Y)


# ## Model Building

# In[70]:


# split into train and test sets 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[71]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[72]:


# build xgboost classifier for multiclassification
xgb_model = XGBClassifier(objective='multi:softprob',eval_metric='auc',num_class=12, use_label_encoder=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)],verbose=0)


# ## Evaluation

# In[73]:


xgb_model.evals_result()


# In[74]:


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


# In[75]:


y_pred = xgb_model.predict(X_test) 
y_pred = [round(value) for value in y_pred]


# In[78]:


# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[79]:


# sensitivity
sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
sensitivity


# In[80]:


# precision
precision = metrics.precision_score(y_test, y_pred, average = 'macro')
precision


# In[81]:


# F1-score
f1 = (2 * precision * sensitivity) / (precision + sensitivity)
f1


# In[82]:


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


# In[83]:


plot_ROC_curve(xgb_model, X_train, y_train, X_test, y_test)


# In[34]:


# normalized confusion matrix


# In[16]:


cm = metrics.confusion_matrix(y_test, y_pred)
result = []
for i in range(12):
    result.append(cm[i] / cm.astype(np.float).sum(axis=1)[i])
normalizeddf1 = pd.DataFrame(result)
normalizeddf1.columns = ['true Drive Alone Free','true Drive Alone Pay','true HOV2 Free',
                       'true HOV2 Pay', 'true HOV3 Free','true HOV3 Pay','true Walk','true BIKE',
                       'true Walk to Transit', 'true Park and Ride','true Kiss and Ride', 'true School Bus']
normalizeddf1.index = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']


# In[17]:


normalizeddf1


# ## Feature Importance

# In[84]:


from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot


# In[85]:


print(xgb_model.feature_importances_)


# In[86]:


pyplot.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
pyplot.show()


# In[87]:


xgb_model.get_booster().feature_names = list(data_encoded.columns)[:-1]
fig, ax = pyplot.subplots(figsize=(5, 10))
plot_importance(xgb_model.get_booster(), ax=ax)
pyplot.show()


# # Feature Selection w/ Imbalanced Sample

# In[161]:


features = ['dempden','oempden','oduden','sovdrivetime','walkttime','ototint','numhouseholdpersons','age','income',            'tolldrivetime','zerototalstops_False','tollcost','walktotransitutility','sovcost','hovdrivetime',            'parkingcost','hovcost']


# In[162]:


combined_subsamples = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)

combined_subsamples


# In[164]:


filtered_subsamples = combined_subsamples[features+['targettripmode']]
filtered_subsamples


# In[165]:


# rearrange into data array for training

data_array = filtered_subsamples.values

X = data_array[:, 0:(data_array.shape[1]-1)]
Y = data_array[:,(data_array.shape[1]-1)]
print(X)
print(Y)


# ## Model Building

# In[166]:


# split into train and test sets 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[167]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[168]:


# build xgboost classifier for multiclassification
xgb_model = XGBClassifier(objective='multi:softprob',eval_metric='auc',num_class=12, use_label_encoder=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)],verbose=0)


# ## Evaluation

# In[169]:


xgb_model.evals_result()


# In[170]:


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


# In[171]:


y_pred = xgb_model.predict(X_test) 
y_pred = [round(value) for value in y_pred]


# In[172]:


# 13*13 confusion matrix
metrics.confusion_matrix(y_test, y_pred)


# In[173]:


# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[174]:


# sensitivity
sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
sensitivity


# In[175]:


# precision
precision = metrics.precision_score(y_test, y_pred, average = 'macro')
precision


# In[176]:


# F1-score
f1 = (2 * precision * sensitivity) / (precision + sensitivity)
f1


# In[177]:


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


# In[178]:


plot_ROC_curve(xgb_model, X_train, y_train, X_test, y_test)


# # Feature Selection w/ Uniform Sample

# In[142]:


# uniform subsample
mode1 = data_encoded[data_encoded['targettripmode']==0].sample(n=9598, random_state=1)
mode2 = data_encoded[data_encoded['targettripmode']==1].sample(n=9598, random_state=1)
mode3 = data_encoded[data_encoded['targettripmode']==2].sample(n=9598, random_state=1)
mode4 = data_encoded[data_encoded['targettripmode']==3].sample(n=9598, random_state=1)
mode5 = data_encoded[data_encoded['targettripmode']==4].sample(n=9598, random_state=1)
mode6 = data_encoded[data_encoded['targettripmode']==5].sample(n=9598, random_state=1)
mode7 = data_encoded[data_encoded['targettripmode']==6].sample(n=9598, random_state=1)
mode8 = data_encoded[data_encoded['targettripmode']==7].sample(n=9598, random_state=1)
mode9 = data_encoded[data_encoded['targettripmode']==8].sample(n=9598, random_state=1)
mode10 = data_encoded[data_encoded['targettripmode']==9].sample(n=9598, random_state=1)
mode11 = data_encoded[data_encoded['targettripmode']==10].sample(n=9598, random_state=1)
mode12 = data_encoded[data_encoded['targettripmode']==11].sample(n=9598, random_state=1)


# In[143]:


combined_subsamples = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)

combined_subsamples


# In[144]:


filtered_subsamples = combined_subsamples[features+['targettripmode']]
filtered_subsamples


# In[145]:


# rearrange into data array for training

data_array = filtered_subsamples.values

X = data_array[:, 0:(data_array.shape[1]-1)]
Y = data_array[:,(data_array.shape[1]-1)]
print(X)
print(Y)


# ## Model Building

# In[146]:


# split into train and test sets 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[147]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[148]:


# build xgboost classifier for multiclassification
xgb_model = XGBClassifier(objective='multi:softprob',eval_metric='auc',num_class=12, use_label_encoder=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)],verbose=0)


# ## Evaluation

# In[149]:


xgb_model.evals_result()


# In[150]:


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


# In[151]:


y_pred = xgb_model.predict(X_test) 
y_pred = [round(value) for value in y_pred]


# In[152]:


# 13*13 confusion matrix
metrics.confusion_matrix(y_test, y_pred)


# In[153]:


# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[154]:


# sensitivity
sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
sensitivity


# In[155]:


# precision
precision = metrics.precision_score(y_test, y_pred, average = 'macro')
precision


# In[156]:


# F1-score
f1 = (2 * precision * sensitivity) / (precision + sensitivity)
f1


# In[157]:


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


# In[158]:


plot_ROC_curve(xgb_model, X_train, y_train, X_test, y_test)


# # All Features w Uniform Sample

# In[179]:


# uniform subsample
mode1 = data_encoded[data_encoded['targettripmode']==0].sample(n=9598, random_state=1)
mode2 = data_encoded[data_encoded['targettripmode']==1].sample(n=9598, random_state=1)
mode3 = data_encoded[data_encoded['targettripmode']==2].sample(n=9598, random_state=1)
mode4 = data_encoded[data_encoded['targettripmode']==3].sample(n=9598, random_state=1)
mode5 = data_encoded[data_encoded['targettripmode']==4].sample(n=9598, random_state=1)
mode6 = data_encoded[data_encoded['targettripmode']==5].sample(n=9598, random_state=1)
mode7 = data_encoded[data_encoded['targettripmode']==6].sample(n=9598, random_state=1)
mode8 = data_encoded[data_encoded['targettripmode']==7].sample(n=9598, random_state=1)
mode9 = data_encoded[data_encoded['targettripmode']==8].sample(n=9598, random_state=1)
mode10 = data_encoded[data_encoded['targettripmode']==9].sample(n=9598, random_state=1)
mode11 = data_encoded[data_encoded['targettripmode']==10].sample(n=9598, random_state=1)
mode12 = data_encoded[data_encoded['targettripmode']==11].sample(n=9598, random_state=1)


# In[180]:


combined_subsamples = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)

combined_subsamples


# In[181]:


# rearrange into data array for training

data_array = combined_subsamples.values

X = data_array[:, 0:(data_array.shape[1]-1)]
Y = data_array[:,(data_array.shape[1]-1)]
print(X)
print(Y)


# ## Model Building

# In[182]:


# split into train and test sets 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[183]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[184]:


# build xgboost classifier for multiclassification
xgb_model = XGBClassifier(objective='multi:softprob',eval_metric='auc',num_class=12, use_label_encoder=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)],verbose=0)


# ## Evaluation

# In[185]:


xgb_model.evals_result()


# In[186]:


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


# In[187]:


y_pred = xgb_model.predict(X_test) 
y_pred = [round(value) for value in y_pred]


# In[188]:


# 13*13 confusion matrix
metrics.confusion_matrix(y_test, y_pred)


# In[189]:


# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[190]:


# sensitivity
sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
sensitivity


# In[191]:


# precision
precision = metrics.precision_score(y_test, y_pred, average = 'macro')
precision


# In[192]:


# F1-score
f1 = (2 * precision * sensitivity) / (precision + sensitivity)
f1


# In[193]:


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


# In[194]:


plot_ROC_curve(xgb_model, X_train, y_train, X_test, y_test)


# # Income Group Evaluation

# In[157]:


middle_list = [x==3 for x in new_utilityvars['income'].values]


# In[158]:


from itertools import compress


# In[159]:


y_pred_middle = list(compress(y_pred, middle_list))
y_test_middle = list(compress(y_test, middle_list))


# In[160]:


# multilabel confusion matrix for middle income group
metrics.multilabel_confusion_matrix(y_test_middle, y_pred_middle)


# In[161]:


# 13*13 confusion matrix
matrix_0 = metrics.confusion_matrix(y_test_middle, y_pred_middle)
matrix_0


# In[162]:


normalized_matrix_middle = metrics.confusion_matrix(y_test_middle, y_pred_middle)/ metrics.confusion_matrix(y_test_middle, y_pred_middle).astype(np.float).sum(axis=1)
normalized_matrix_middle


# In[163]:


normalizeddf_middle = pd.DataFrame(
    metrics.confusion_matrix(y_test_middle, y_pred_middle)/ metrics.confusion_matrix(y_test_middle, y_pred_middle).astype(np.float).sum(axis=1))


# In[164]:


normalizeddf_middle.columns = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']
normalizeddf_middle.index = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']


# In[165]:


normalizeddf_middle


# In[169]:


accuracy_df_middle = pd.DataFrame(normalized_matrix_middle.diagonal()/normalized_matrix_middle.sum(axis=1))


# In[170]:


accuracy_df_middle.index = travel_names


# In[171]:


accuracy_df_middle.plot(kind='bar')


# In[175]:


low_list = [x==1 or x==2 for x in new_utilityvars['income'].values]


# In[176]:


y_pred_low = list(compress(y_pred, low_list))
y_test_low = list(compress(y_test, low_list))


# In[177]:


metrics.multilabel_confusion_matrix(y_test_low, y_pred_low)


# In[178]:


# 13*13 confusion matrix
matrix_0 = metrics.confusion_matrix(y_test_low, y_pred_low)
matrix_0


# In[179]:


normalized_matrix_low = metrics.confusion_matrix(y_test_low, y_pred_low)/ metrics.confusion_matrix(y_test_low, y_pred_low).astype(np.float).sum(axis=1)
normalized_matrix_low


# In[180]:


normalizeddf_low = pd.DataFrame(
    metrics.confusion_matrix(y_test_low, y_pred_low)/ metrics.confusion_matrix(y_test_low, y_pred_low).astype(np.float).sum(axis=1))


# In[181]:


normalizeddf_low.columns = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']
normalizeddf_low.index = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']


# In[182]:


normalizeddf_low


# In[183]:


accuracy_df_low = pd.DataFrame(normalized_matrix_low.diagonal()/normalized_matrix_low.sum(axis=1))
accuracy_df_low.index = travel_names


# In[184]:


accuracy_df_low.plot(kind='bar')


# In[185]:


# for high income
high_list = [x==4 or x==5 for x in new_utilityvars['income'].values]


# In[186]:


y_pred_high = list(compress(y_pred, high_list))
y_test_high = list(compress(y_test, high_list))


# In[187]:


metrics.multilabel_confusion_matrix(y_test_high, y_pred_high)
# 13*13 confusion matrix
matrix_0 = metrics.confusion_matrix(y_test_high, y_pred_high)
matrix_0


# In[188]:


normalized_matrix_high = metrics.confusion_matrix(y_test_high, y_pred_high)/ metrics.confusion_matrix(y_test_high, y_pred_high).astype(np.float).sum(axis=1)
normalized_matrix_high


# In[189]:


normalizeddf_high = pd.DataFrame(
    metrics.confusion_matrix(y_test_high, y_pred_high)/ metrics.confusion_matrix(y_test_high, y_pred_high).astype(np.float).sum(axis=1))

normalizeddf_high.columns = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']
normalizeddf_high.index = ['pred Drive Alone Free','pred Drive Alone Pay','pred HOV2 Free',
                       'pred HOV2 Pay', 'pred HOV3 Free','pred HOV3 Pay','pred Walk','pred BIKE',
                       'pred Walk to Transit', 'pred Park and Ride','pred Kiss and Ride', 'pred School Bus']


# In[190]:


normalizeddf_high


# In[191]:


accuracy_df_high = pd.DataFrame(normalized_matrix_high.diagonal()/normalized_matrix_high.sum(axis=1))
accuracy_df_high.index = travel_names


# In[192]:


accuracy_df_high.plot(kind='bar')


# In[ ]:




