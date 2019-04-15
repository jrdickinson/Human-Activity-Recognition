#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pandas as pd
import numpy as np
from collections import Counter
os.chdir("UCI HAR Dataset/UCI HAR Dataset/test")
x_test = pd.read_csv("X_test.txt", "\s+", header=None)
y_test = pd.read_csv("y_test.txt", "\s+", header=None)
os.chdir("UCI HAR Dataset/UCI HAR Dataset/train")
x_train = pd.read_csv("X_train.txt", "\s+", header=None)
y_train = pd.read_csv("y_train.txt", "\s+", header=None)


# In[99]:


print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)


# ### Normalize values

# In[18]:


xmin = x_train.min()
xmax = x_train.max()
x_train = x_train-xmin/(xmax-xmin)
x_test = x_test-xmin/(xmax-xmin)


# ### View individuals

# In[19]:


os.chdir("UCI HAR Dataset/UCI HAR Dataset/test")
text = pd.read_csv("subject_test.txt")
# text.iloc[617]
# text[text["2"]==4]
text.iloc[0:301]["2"].unique()
text.iloc[618:906]["2"].unique()
text.iloc[301:618]["2"].unique()
text.iloc[906:1200]["2"].unique()
text.iloc[1200:1520]["2"].unique()
text.iloc[1520:1847]["2"].unique()
text.iloc[1847:2211]["2"].unique()
text.iloc[2211:2565]["2"].unique()
text.iloc[2565:2946]["2"].unique()


# ### Check Class Balance

# In[102]:


print(len(y_train[y_train[0]==1]))
print(len(y_train[y_train[0]==2]))
print(len(y_train[y_train[0]==3]))
print(len(y_train[y_train[0]==4]))
print(len(y_train[y_train[0]==5]))
print(len(y_train[y_train[0]==6]))
print(len(y_test[y_test[0]==1]))
print(len(y_test[y_test[0]==2]))
print(len(y_test[y_test[0]==3]))
print(len(y_test[y_test[0]==4]))
print(len(y_test[y_test[0]==5]))
print(len(y_test[y_test[0]==6]))


# ### Create Validation Set

# In[20]:


x_valid = x_test.iloc[0:1520]
x_test = x_test.iloc[1520:2946]
y_valid = y_test.iloc[0:1520]
y_test = y_test.iloc[1520:2946]


# ### Iteratively drop columns XGBoost

# In[67]:


# # First Iteration
# select_columns = np.r_[0:561]
# # Second Iteration
# select_columns = np.r_[0:19,20:28,29:39,40,42:50,55:60,61:94,95:108,113,114,120,123,124,126,131:137,140,141,143:150,151:154,156,157,159:163,165:168,170,172:177,182:186,187:195,196:204,210,213,215:221,224,245,246,248,249,255,263,271,282,300,301,318,319,321,322,325,330,331,333,342,344,347,348,350,354:357,369,372:375,388:392,398:406,407,408,410,413,417:422,426,427,429:444,447,448,450,452:458,459:467,468:473,474:513,514:527,529:533,534,535,538:561]
# # Third Iteration
select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
# # Fifth Iteration
select_columns = select_columns[np.r_[0,1,3:10,11:14,15,16,18,19,23,24,26,28:31,32,34:46,48,51,53:59,62:66,67,69:72,75:78,82,86,107]]
x_train_temp = x_train.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import xgboost
gridXG = ParameterGrid({"n_estimators": [10],
                        "learning_rate": [.8],
                        "max_depth": [4],
                        "booster": ["dart"], #"gbtree", "gblinear"
                        "subsample": [.6],
                        "colsample_bytree": [.8],
                        "colsample_bylevel": [1],
                        "n_jobs": [4],
                        "gamma": [0],
                        "min_child_weight": [.4],
                        "max_delta_step": [4],
                        "colsample_bynode": [.9],
                        "reg_alpha": [.1],
                        "random_state": [5,15,35,55,71,88,104,112,128,161]
                       })
nums = []
for i in range(0,561):
    print(i)
    x_train_sub = x_train_temp.drop(x_train_temp.columns[i], axis=1)
    x_valid_sub = x_valid_temp.drop(x_valid_temp.columns[i], axis=1)
    for j, params in enumerate(gridXG):
        start = time.time()
        model = xgboost.XGBClassifier(**params)
        model.fit(x_train_sub, y_train)
        score = model.predict(x_valid_sub)
        cm = confusion_matrix(y_valid, score)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("accuracy_full: {}".format(sum(cm.diagonal())/6))
        precision, recall, fscore, support = precision_recall_fscore_support(y_valid, score)
        print('precision_full: {}'.format(precision))
        print('recall_full: {}'.format(recall))
        end = time.time()
        print(end-start)
        if sum(cm.diagonal())/6<.935:
            nums.append(str(i)+" - "+str(j))


# ### Test results and parameters on validation XGBoost

# In[68]:


# # First Iteration
# select_columns = np.r_[0:561]
# # Second Iteration
# select_columns = np.r_[0:19,20:28,29:39,40,42:50,55:60,61:94,95:108,113,114,120,123,124,126,131:137,140,141,143:150,151:154,156,157,159:163,165:168,170,172:177,182:186,187:195,196:204,210,213,215:221,224,245,246,248,249,255,263,271,282,300,301,318,319,321,322,325,330,331,333,342,344,347,348,350,354:357,369,372:375,388:392,398:406,407,408,410,413,417:422,426,427,429:444,447,448,450,452:458,459:467,468:473,474:513,514:527,529:533,534,535,538:561]
# # Third Iteration
select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
# # Fifth Iteration
# select_columns = select_columns[np.r_[0,1,3:10,11:14,15,16,18,19,23,24,26,28:31,32,34:46,48,51,53:59,62:66,67,69:72,75:78,82,86,107]]
# # # Sixth Iteration
# select_columns = select_columns[np.r_[3,4,6,8:14,18:21,23:26,28,30,34,36,40,41,45,46,48:51,53,54,56]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import xgboost
gridXG = ParameterGrid({"n_estimators": [120],
                        "learning_rate": [.1],
                        "max_depth": [5],
                        "booster": ["dart"], #"gbtree", "gblinear"
                        "subsample": [.2],
                        "colsample_bytree": [1],
                        "colsample_bylevel": [1],
                        "n_jobs": [4],
                        "gamma": [0],
                        "min_child_weight": [.1],
                        "max_delta_step": [4],
                        "colsample_bynode": [1],
                        "reg_alpha": [0], 
                        "random_state": [1]
                       })

for i, params in enumerate(gridXG):
    start = time.time()
    model = xgboost.XGBClassifier(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_valid_temp)
    cm = confusion_matrix(y_valid, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    precision, recall, fscore, support = precision_recall_fscore_support(y_valid, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(i, params)
    print(end-start)
    


# In[71]:


iteration = []
seed = []
for i in nums:
    iteration.append(int(i.split(" - ")[0]))
    seed.append(int(i.split(" - ")[1]))


# In[49]:


# Counter(iteration)


# ### Final Test XGBoost

# In[72]:


select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import xgboost
gridXG = ParameterGrid({"n_estimators": [120],
                        "learning_rate": [.1],
                        "max_depth": [5],
                        "booster": ["dart"], #"gbtree", "gblinear"
                        "subsample": [.2],
                        "colsample_bytree": [1],
                        "colsample_bylevel": [1],
                        "n_jobs": [4],
                        "gamma": [0],
                        "min_child_weight": [.1],
                        "max_delta_step": [4],
                        "colsample_bynode": [1],
                        "reg_alpha": [0], 
                        "random_state": [210,220,230,240,250,260,270,280,290,300]
                       })

for params in gridXG:
    start = time.time()
    model = xgboost.XGBClassifier(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_test_temp)
    cm = confusion_matrix(y_test, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    print("accuracy_full: {}".format(cm.diagonal()))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(end-start)


# ### Test results and parameters on validation Adaboost

# In[74]:


# # First Iteration
# select_columns = np.r_[0:561]
# # Second Iteration
# select_columns = np.r_[0:19,20:28,29:39,40,42:50,55:60,61:94,95:108,113,114,120,123,124,126,131:137,140,141,143:150,151:154,156,157,159:163,165:168,170,172:177,182:186,187:195,196:204,210,213,215:221,224,245,246,248,249,255,263,271,282,300,301,318,319,321,322,325,330,331,333,342,344,347,348,350,354:357,369,372:375,388:392,398:406,407,408,410,413,417:422,426,427,429:444,447,448,450,452:458,459:467,468:473,474:513,514:527,529:533,534,535,538:561]
# # Third Iteration
select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
# # Fifth Iteration
# select_columns = select_columns[np.r_[0,1,3:10,11:14,15,16,18,19,23,24,26,28:31,32,34:46,48,51,53:59,62:66,67,69:72,75:78,82,86,107]]
# # # Sixth Iteration
# select_columns = select_columns[np.r_[3,4,6,8:14,18:21,23:26,28,30,34,36,40,41,45,46,48:51,53,54,56]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier
gridAB = ParameterGrid({"n_estimators": [160],
                        "learning_rate": [.8],
                        })

for i, params in enumerate(gridAB):
    start = time.time()
    model = AdaBoostClassifier(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_valid_temp)
    cm = confusion_matrix(y_valid, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    precision, recall, fscore, support = precision_recall_fscore_support(y_valid, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(i, params)
    print(end-start)
    


# ### Final Test Adaboost

# In[75]:


select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier
gridAB = ParameterGrid({"n_estimators": [160],
                        "learning_rate": [.8]
                        })

for params in gridAB:
    start = time.time()
    model = AdaBoostClassifier(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_test_temp)
    cm = confusion_matrix(y_test, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    print("accuracy_full: {}".format(cm.diagonal()))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(end-start)


# ### Test results and parameters on validation Random Forest

# In[21]:


# # First Iteration
# select_columns = np.r_[0:561]
# # Second Iteration
# select_columns = np.r_[0:19,20:28,29:39,40,42:50,55:60,61:94,95:108,113,114,120,123,124,126,131:137,140,141,143:150,151:154,156,157,159:163,165:168,170,172:177,182:186,187:195,196:204,210,213,215:221,224,245,246,248,249,255,263,271,282,300,301,318,319,321,322,325,330,331,333,342,344,347,348,350,354:357,369,372:375,388:392,398:406,407,408,410,413,417:422,426,427,429:444,447,448,450,452:458,459:467,468:473,474:513,514:527,529:533,534,535,538:561]
# # Third Iteration
select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
# # Fifth Iteration
# select_columns = select_columns[np.r_[0,1,3:10,11:14,15,16,18,19,23,24,26,28:31,32,34:46,48,51,53:59,62:66,67,69:72,75:78,82,86,107]]
# # # Sixth Iteration
# select_columns = select_columns[np.r_[3,4,6,8:14,18:21,23:26,28,30,34,36,40,41,45,46,48:51,53,54,56]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
gridRF = ParameterGrid({"n_estimators": [120],
                        "min_samples_split": [.01],
                        "max_features": [70],
                        "max_depth": [9],
                        "min_samples_leaf": [13],
                        "bootstrap": ["True"],
                        "max_leaf_nodes": [50],
                        "n_jobs": [4],
                        })

for i, params in enumerate(gridRF):
    start = time.time()
    model = RandomForestClassifier(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_valid_temp)
    cm = confusion_matrix(y_valid, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    precision, recall, fscore, support = precision_recall_fscore_support(y_valid, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(i, params)
    print(end-start)
    


# ### Final Test Random Forest

# In[112]:


select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
gridRF = ParameterGrid({"n_estimators": [120],
                        "min_samples_split": [.01],
                        "max_features": [70],
                        "max_depth": [9],
                        "min_samples_leaf": [13],
                        "bootstrap": ["True"],
                        "max_leaf_nodes": [50],
                        "n_jobs": [4],
                        })

for params in gridRF:
    start = time.time()
    model = RandomForestClassifier(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_test_temp)
    cm = confusion_matrix(y_test, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    print("accuracy_full: {}".format(cm.diagonal()))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(end-start)


# ### Test results and parameters on validation SVM

# In[4]:


#I trained on a different file with train and validation combined.  Results in the final SVM cell appear a little inflated.

# # First Iteration
select_columns = np.r_[0:561]
# # Second Iteration
# select_columns = np.r_[0:19,20:28,29:39,40,42:50,55:60,61:94,95:108,113,114,120,123,124,126,131:137,140,141,143:150,151:154,156,157,159:163,165:168,170,172:177,182:186,187:195,196:204,210,213,215:221,224,245,246,248,249,255,263,271,282,300,301,318,319,321,322,325,330,331,333,342,344,347,348,350,354:357,369,372:375,388:392,398:406,407,408,410,413,417:422,426,427,429:444,447,448,450,452:458,459:467,468:473,474:513,514:527,529:533,534,535,538:561]
# # Third Iteration
# select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # # Fourth Iteration
# select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
# # Fifth Iteration
# select_columns = select_columns[np.r_[0,1,3:10,11:14,15,16,18,19,23,24,26,28:31,32,34:46,48,51,53:59,62:66,67,69:72,75:78,82,86,107]]
# # # Sixth Iteration
# select_columns = select_columns[np.r_[3,4,6,8:14,18:21,23:26,28,30,34,36,40,41,45,46,48:51,53,54,56]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC  
gridSVC = ParameterGrid({"kernel": ["linear"], #, "poly", "rbf", "sigmoid"
                         "gamma": [.1],
                         "degree": [6],
                         "C": [3.5],
#                          "tol": [1],
#                          "random_state": [1],
                         "max_iter": [1000000]
                       })

for i, params in enumerate(gridSVC):
    start = time.time()
    model = SVC(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_valid_temp)
    cm = confusion_matrix(y_valid, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    precision, recall, fscore, support = precision_recall_fscore_support(y_valid, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(i, params)
    print(end-start)
    


# ### Final Test SVM

# In[5]:


select_columns = np.r_[0:561]
# select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # # Fourth Iteration
# select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
x_train_temp = x_train.iloc[:,select_columns]
x_valid_temp = x_valid.iloc[:,select_columns]
x_test_temp = x_test.iloc[:,select_columns]
from sklearn.model_selection import ParameterGrid
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC  
gridSVC = ParameterGrid({"kernel": ["linear"], #, "poly", "rbf", "sigmoid"
                         "gamma": [.1],
                         "degree": [6],
                         "C": [3.5],
#                          "tol": [1],
#                          "random_state": [1],
                         "max_iter": [1000000]
                       })

for params in gridSVC:
    start = time.time()
    model = SVC(**params)
    model.fit(x_train_temp, y_train)
    score = model.predict(x_test_temp)
    cm = confusion_matrix(y_test, score)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy_full: {}".format(sum(cm.diagonal())/6))
    print("accuracy_full: {}".format(cm.diagonal()))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, score)
    print('precision_full: {}'.format(precision))
    print('recall_full: {}'.format(recall))
    end = time.time()
    print(end-start)


# ### Read in files

# In[13]:


import os
import pandas as pd
import numpy as np
from collections import Counter
os.chdir("UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals")
test1 = pd.read_csv("body_acc_x_test.txt", "\s+", header=None)
test2 = pd.read_csv("body_acc_y_test.txt", "\s+", header=None)
test3 = pd.read_csv("body_acc_z_test.txt", "\s+", header=None)
test4 = pd.read_csv("body_gyro_x_test.txt", "\s+", header=None)
test5 = pd.read_csv("body_gyro_y_test.txt", "\s+", header=None)
test6 = pd.read_csv("body_gyro_z_test.txt", "\s+", header=None)
test7 = pd.read_csv("total_acc_x_test.txt", "\s+", header=None)
test8 = pd.read_csv("total_acc_y_test.txt", "\s+", header=None)
test9 = pd.read_csv("total_acc_z_test.txt", "\s+", header=None)
os.chdir("UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals")
train1 = pd.read_csv("body_acc_x_train.txt", "\s+", header=None)
train2 = pd.read_csv("body_acc_y_train.txt", "\s+", header=None)
train3 = pd.read_csv("body_acc_z_train.txt", "\s+", header=None)
train4 = pd.read_csv("body_gyro_x_train.txt", "\s+", header=None)
train5 = pd.read_csv("body_gyro_y_train.txt", "\s+", header=None)
train6 = pd.read_csv("body_gyro_z_train.txt", "\s+", header=None)
train7 = pd.read_csv("total_acc_x_train.txt", "\s+", header=None)
train8 = pd.read_csv("total_acc_y_train.txt", "\s+", header=None)
train9 = pd.read_csv("total_acc_z_train.txt", "\s+", header=None)


# ### Create arrays in correct shape

# In[14]:


x_test = np.dstack((test1,test2,test3,test4,test5,test6,test7,test8,test9))
x_train = np.dstack((train1,train2,train3,train4,train5,train6,train7,train8,train9))
os.chdir("UCI HAR Dataset/UCI HAR Dataset/test")
y_test = pd.read_csv("y_test.txt", "\s+", header=None)
os.chdir("UCI HAR Dataset/UCI HAR Dataset/train")
y_train = pd.read_csv("y_train.txt", "\s+", header=None)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[15]:


y_train = pd.get_dummies(y_train[0]).values
y_test = pd.get_dummies(y_test[0]).values
x_train = x_train.reshape([7352,1152])
x_test = x_test.reshape([2947,1152])
print(y_test.shape)
y_test


# In[83]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import pandas as pd
from skimage import io
from PIL import Image
import functools
from matplotlib import pyplot


# ### Run ANN

# In[84]:


from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import MobileNet
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Flatten, Dense, LeakyReLU, Dropout, GaussianNoise
from tensorflow import set_random_seed
from keras.optimizers import Adam, RMSprop
from keras_gradient_noise import add_gradient_noise
from keras.initializers import glorot_uniform, glorot_normal, he_uniform, he_normal
set_random_seed(1724)


model = models.Sequential()
model.add(Dense(128, kernel_initializer="glorot_normal", input_shape=(9*128,)))
model.add(LeakyReLU())
model.add(Dropout(.15))
model.add(BatchNormalization())
# model.add(GaussianNoise(1))
model.add(Dense(64))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dense(6, activation='softmax'))


# In[85]:


# Let's use a different optimizer this time
noisy = add_gradient_noise(RMSprop)
model.compile(optimizer="Adamax",
# model.compile(optimizer=noisy(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# In[86]:


callbacks = [EarlyStopping(monitor='val_loss', patience=16),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# In[87]:


history = model.fit(x_train, y_train, epochs=250, batch_size=16, validation_data = (x_test, y_test), callbacks=callbacks)
from keras.models import load_model
# model = load_model('best_model.h5', custom_objects={"NoisyRMSprop":noisy()})
model = load_model('best_model.h5')
model.evaluate(x_test, y_test)


# ### Try ANN again with first 64 of 128 time values per row

# In[88]:


x_test = np.dstack((test1.iloc[:,0:64],test2.iloc[:,0:64],test3.iloc[:,0:64],test4.iloc[:,0:64],test5.iloc[:,0:64],test6.iloc[:,0:64],test7.iloc[:,0:64],test8.iloc[:,0:64],test9.iloc[:,0:64]))
x_train = np.dstack((train1.iloc[:,0:64],train2.iloc[:,0:64],train3.iloc[:,0:64],train4.iloc[:,0:64],train5.iloc[:,0:64],train6.iloc[:,0:64],train7.iloc[:,0:64],train8.iloc[:,0:64],train9.iloc[:,0:64]))
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape([7352,576])
x_test = x_test.reshape([2947,576])


# In[89]:


model = models.Sequential()
model.add(Dense(1024, kernel_initializer="glorot_normal", input_shape=(9*64,)))
model.add(LeakyReLU())
model.add(Dropout(.15))
model.add(BatchNormalization())
# model.add(GaussianNoise(1))
model.add(Dense(64))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dense(6, activation='softmax'))


# In[90]:


# Let's use a different optimizer this time
noisy = add_gradient_noise(RMSprop)
model.compile(optimizer="Adamax",
# model.compile(optimizer=noisy(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# In[91]:


callbacks = [EarlyStopping(monitor='val_loss', patience=16),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# In[92]:


history = model.fit(x_train, y_train, epochs=250, batch_size=16, validation_data = (x_test, y_test), callbacks=callbacks)
from keras.models import load_model
# model = load_model('best_model.h5', custom_objects={"NoisyRMSprop":noisy()})
model = load_model('best_model.h5')
model.evaluate(x_test, y_test)


# ### Import data for LSTM

# In[1]:


import os
import pandas as pd
import numpy as np
from collections import Counter
os.chdir("UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals")
test1 = pd.read_csv("body_acc_x_test.txt", "\s+", header=None)
test2 = pd.read_csv("body_acc_y_test.txt", "\s+", header=None)
test3 = pd.read_csv("body_acc_z_test.txt", "\s+", header=None)
test4 = pd.read_csv("body_gyro_x_test.txt", "\s+", header=None)
test5 = pd.read_csv("body_gyro_y_test.txt", "\s+", header=None)
test6 = pd.read_csv("body_gyro_z_test.txt", "\s+", header=None)
test7 = pd.read_csv("total_acc_x_test.txt", "\s+", header=None)
test8 = pd.read_csv("total_acc_y_test.txt", "\s+", header=None)
test9 = pd.read_csv("total_acc_z_test.txt", "\s+", header=None)
os.chdir("UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals")
train1 = pd.read_csv("body_acc_x_train.txt", "\s+", header=None)
train2 = pd.read_csv("body_acc_y_train.txt", "\s+", header=None)
train3 = pd.read_csv("body_acc_z_train.txt", "\s+", header=None)
train4 = pd.read_csv("body_gyro_x_train.txt", "\s+", header=None)
train5 = pd.read_csv("body_gyro_y_train.txt", "\s+", header=None)
train6 = pd.read_csv("body_gyro_z_train.txt", "\s+", header=None)
train7 = pd.read_csv("total_acc_x_train.txt", "\s+", header=None)
train8 = pd.read_csv("total_acc_y_train.txt", "\s+", header=None)
train9 = pd.read_csv("total_acc_z_train.txt", "\s+", header=None)


# In[2]:


x_test = np.dstack((test1,test2,test3,test4,test5,test6,test7,test8,test9))
x_train = np.dstack((train1,train2,train3,train4,train5,train6,train7,train8,train9))
os.chdir("UCI HAR Dataset/UCI HAR Dataset/test")
y_test = pd.read_csv("y_test.txt", "\s+", header=None)
os.chdir("UCI HAR Dataset/UCI HAR Dataset/train")
y_train = pd.read_csv("y_train.txt", "\s+", header=None)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### One-hot encode y

# In[3]:


y_train = pd.get_dummies(y_train[0]).values
y_test = pd.get_dummies(y_test[0]).values
# x_train = x_train.reshape([7352,1152])
# x_test = x_test.reshape([2947,1152])
print(y_test.shape)
y_test


# ### LSTM

# In[4]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import pandas as pd
from skimage import io
from PIL import Image
import functools
from matplotlib import pyplot


# In[5]:


from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import MobileNet
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Flatten, Dense, LeakyReLU, Dropout, GaussianNoise
from tensorflow import set_random_seed
from keras.optimizers import Adam, RMSprop
from keras_gradient_noise import add_gradient_noise
from keras.initializers import glorot_uniform, glorot_normal, he_uniform, he_normal
set_random_seed(1724)


model = models.Sequential()
model.add(LSTM(100, input_shape=(128,9)))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer="glorot_normal", input_shape=(9*128,)))
model.add(LeakyReLU())
model.add(Dropout(.15))
model.add(BatchNormalization())
# model.add(GaussianNoise(1))
model.add(Dense(64))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dense(6, activation='softmax'))


# In[6]:


# Let's use a different optimizer this time
noisy = add_gradient_noise(Adam)
model.compile(optimizer="adam",
# model.compile(optimizer=noisy(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# In[7]:


callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


# In[ ]:


history = model.fit(x_train, y_train, epochs=250, batch_size=32, validation_data = (x_test, y_test), callbacks=callbacks)
from keras.models import load_model
# model = load_model('best_model.h5', custom_objects={"NoisyAdam":noisy()})
model = load_model('best_model.h5')
model.evaluate(x_test, y_test)

