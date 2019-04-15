
# Human Activity Recognition

### This dataset contains information collected from a UCI experiment designed to detect activity.  30 individuals participated in the experiment wearing sensors that output accelerometer and gyroscope signals.  The six activities that are labeled in the dataset are walking, walking upstairs, walking downstairs, sitting, standing, and laying.  More technical details can be found on the UCI site.

###  This dataset comes with pre-cleaned data for x_train, y_train, x_test, and y_test.  We'll read those files in and see what they look like.


```python
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
print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)
```

    (2947, 561)
    (2947, 1)
    (7352, 561)
    (7352, 1)
    

### Next, we'll normalize the values before we start trying our machine learning methods


```python
xmin = x_train.min()
xmax = x_train.max()
x_train = x_train-xmin/(xmax-xmin)
x_test = x_test-xmin/(xmax-xmin)
```

### One last thing we want to do is check that there are no imbalances in the classes that we'll need to deal with.  We also want to check for imbalance between individuals.  This will be useful since we will be splitting the test into both a test and validation set.


```python
os.chdir("UCI HAR Dataset/UCI HAR Dataset/test")
text = pd.read_csv("subject_test.txt")
text.iloc[0:301]["2"].unique()
text.iloc[618:906]["2"].unique()
text.iloc[301:618]["2"].unique()
text.iloc[906:1200]["2"].unique()
text.iloc[1200:1520]["2"].unique()
text.iloc[1520:1847]["2"].unique()
text.iloc[1847:2211]["2"].unique()
text.iloc[2211:2565]["2"].unique()
text.iloc[2565:2946]["2"].unique()
```




    array([24], dtype=int64)




```python
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
```

    1226
    1073
    986
    1286
    1374
    1407
    496
    471
    420
    491
    532
    537
    

### As we can see, there are some difference, but groups are similar enough that we won't have to do anything to fix them.  Now we'll create the validation set and then we can start trying our machine learning methods.


```python
x_valid = x_test.iloc[0:1520]
x_test = x_test.iloc[1520:2946]
y_valid = y_test.iloc[0:1520]
y_test = y_test.iloc[1520:2946]
```

### First we'll start with XGBoost since tends to make strong models.  I'm going to do a bit of a lengthy procedure here to see how well XGBoost will perform.  I run through the model multiple times iteratively dropping 1 column.  Columns that have lower accuracies are stronger features.  A cutoff point will be created to remove the weaker features from the model.  Then the parameters will be retuned and we'll repeat the process until we find the feature set that seems like it will do best.  Number of iterations and max depth will be kept low for these parts since those features really increase the amount of computing time, but when we're satisfied with the feature selection, we'll tune those as well.  Finally, we'll test on the test set to see how well the model performed.


```python
# # First Iteration
# select_columns = np.r_[0:561]
# # Second Iteration
# select_columns = np.r_[0:19,20:28,29:39,40,42:50,55:60,61:94,95:108,113,114,120,123,124,126,131:137,140,141,143:150,151:154,156,157,159:163,165:168,170,172:177,182:186,187:195,196:204,210,213,215:221,224,245,246,248,249,255,263,271,282,300,301,318,319,321,322,325,330,331,333,342,344,347,348,350,354:357,369,372:375,388:392,398:406,407,408,410,413,417:422,426,427,429:444,447,448,450,452:458,459:467,468:473,474:513,514:527,529:533,534,535,538:561]
# # Third Iteration
select_columns = np.r_[0,1,4,5,9,11,21,23,38,42,43,45,46,47,48,49,55,56,58,59,61,62,64,65,67,68,69,71,74,75,76,80,81,83,84,89,90,92,93,95,99,100,101,102,103,114,120,126,131,132,134,145,151,152,153,157,176,183,187,188,189,193,194,197,199,202,203,210,213,215,216,218,245,246,248,249,255,263,271,319,330,333,342,350,355,356,388,407,408,410,429,430,432,433,434,435,437,438,439,440,441,442,443,447,448,450,452,453,454,455,456,457,459,460,461,462,463,464,465,466,468,469,472,474,479,486,487,488,491,496,498,499,503,504,506,507,508,509,511,512,514,515,516,517,518,519,520,521,523,524,525,538,539,540,541,542,543,544,548,555,557,558]
# # Fourth Iteration
select_columns = select_columns[np.r_[10,11,13,14,17:22,23:28,29:35,36,37,39:47,49,50,52:56,59:65,67:74,75,79,91:94,95,99,104:109,110:120,121:137,138,140:147,148:154,155:162]]
# # # Fifth Iteration
# select_columns = select_columns[np.r_[0,1,3:10,11:14,15,16,18,19,23,24,26,28:31,32,34:46,48,51,53:59,62:66,67,69:72,75:78,82,86,107]]
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

```


```python
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
    
```

    accuracy_full: 0.9681184312015887
    precision_full: [0.97857143 0.99585062 0.98173516 0.93469388 0.92075472 1.        ]
    recall_full: [1.         0.97165992 0.98623853 0.9123506  0.93846154 1.        ]
    0 {'booster': 'dart', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 4, 'max_depth': 5, 'min_child_weight': 0.1, 'n_estimators': 120, 'n_jobs': 4, 'random_state': 1, 'reg_alpha': 0, 'subsample': 0.2}
    40.1768639087677
    


```python
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
                        "random_state": [1]
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
```

    accuracy_full: 0.9447806535423018
    accuracy_full: [0.96396396 1.         0.85643564 0.86666667 0.98161765 1.        ]
    precision_full: [0.97716895 0.89558233 0.96648045 0.97652582 0.89297659 1.        ]
    recall_full: [0.96396396 1.         0.85643564 0.86666667 0.98161765 1.        ]
    33.41908288002014
    

### We can see that XGBoost performed very well on this set.  We'll try a few other models and see how they do.

### Test results and parameters on validation Adaboost


```python
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
    
```

    C:\Users\jerem.DESKTOP-GGM6Q2I\Anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    accuracy_full: 0.691961569679297
    precision_full: [0.93714286 0.85820896 0.49767442 0.62886598 0.55188679 1.        ]
    recall_full: [0.59854015 0.46558704 0.98165138 0.24302789 0.9        0.96296296]
    0 {'learning_rate': 0.8, 'n_estimators': 160}
    16.50082039833069
    

### Final Test Adaboost


```python
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
```

    C:\Users\jerem.DESKTOP-GGM6Q2I\Anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    accuracy_full: 0.7083252242504541
    accuracy_full: [0.54954955 0.69058296 0.83663366 0.225      0.96691176 0.98127341]
    precision_full: [0.89051095 0.79381443 0.53481013 0.79411765 0.5857461  1.        ]
    recall_full: [0.54954955 0.69058296 0.83663366 0.225      0.96691176 0.98127341]
    15.902435064315796
    

### Test results and parameters on validation Random Forest


```python
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
    
```

    C:\Users\jerem.DESKTOP-GGM6Q2I\Anaconda3\lib\site-packages\ipykernel_launcher.py:36: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    

    accuracy_full: 0.9192378623842378
    precision_full: [0.87055016 0.99514563 0.93777778 0.91479821 0.83623693 1.        ]
    recall_full: [0.98175182 0.82995951 0.96788991 0.812749   0.92307692 1.        ]
    0 {'bootstrap': 'True', 'max_depth': 9, 'max_features': 70, 'max_leaf_nodes': 50, 'min_samples_leaf': 13, 'min_samples_split': 0.01, 'n_estimators': 120, 'n_jobs': 4}
    19.42812418937683
    

### Final Test Random Forest


```python
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
```

    C:\Users\jerem.DESKTOP-GGM6Q2I\Anaconda3\lib\site-packages\ipykernel_launcher.py:27: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    

    accuracy_full: 0.8965247410108068
    accuracy_full: [0.92342342 0.97757848 0.7029703  0.80833333 0.97058824 0.99625468]
    precision_full: [0.93181818 0.77857143 0.96598639 0.95566502 0.8516129  1.        ]
    recall_full: [0.92342342 0.97757848 0.7029703  0.80833333 0.97058824 0.99625468]
    22.77300500869751
    

### Test results and parameters on validation SVM


```python
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
    
```

    C:\Users\jerem.DESKTOP-GGM6Q2I\Anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    accuracy_full: 0.9404100047806412
    precision_full: [0.9375     0.98275862 0.97747748 0.9468599  0.82724252 1.        ]
    recall_full: [0.98540146 0.92307692 0.99541284 0.78087649 0.95769231 1.        ]
    0 {'C': 3.5, 'degree': 6, 'gamma': 0.1, 'kernel': 'linear', 'max_iter': 1000000, 'random_state': 1}
    3.3948216438293457
    

### Final Test SVM


```python
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
```

    C:\Users\jerem.DESKTOP-GGM6Q2I\Anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    accuracy_full: 0.9878236052964304
    accuracy_full: [1.         0.9955157  0.95544554 0.98333333 0.99264706 1.        ]
    precision_full: [0.98230088 0.97368421 1.         0.99159664 0.98540146 1.        ]
    recall_full: [1.         0.9955157  0.95544554 0.98333333 0.99264706 1.        ]
    3.241715669631958
    

### SVM was another very strong model and random forest also did fairly well.  The test results shown for SVM are a little inflated since those parameters were tested seperately without a validation set.  Now that we saw how the machine learning models performed, the larger datasets will be used to create neural network models.


```python
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
```


```python
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
```

    (7352, 128, 9)
    (2947, 128, 9)
    (7352, 1)
    (2947, 1)
    

### Reshape x and one-hot encode y


```python
y_train = pd.get_dummies(y_train[0]).values
y_test = pd.get_dummies(y_test[0]).values
x_train = x_train.reshape([7352,1152])
x_test = x_test.reshape([2947,1152])
print(y_test.shape)
y_test
```

    (2947, 6)
    




    array([[0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0],
           ...,
           [0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0]], dtype=uint8)



### Now that we have the correct shapes, we'll experiment with several ways to optimize the model like adding noise to the inputs and the gradient, weight initializers, dropout, batch normalization, width and depth of the neural network, and different activation functions and optimizers. 

### Run ANN


```python
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

```


```python
# Let's use a different optimizer this time
noisy = add_gradient_noise(RMSprop)
model.compile(optimizer="Adamax",
# model.compile(optimizer=noisy(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```


```python
callbacks = [EarlyStopping(monitor='val_loss', patience=16),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
```


```python
history = model.fit(x_train, y_train, epochs=250, batch_size=16, validation_data = (x_test, y_test), callbacks=callbacks)
from keras.models import load_model
# model = load_model('best_model.h5', custom_objects={"NoisyRMSprop":noisy()})
model = load_model('best_model.h5')
model.evaluate(x_test, y_test)
```

    2947/2947 [==============================] - 0s 113us/step
    




    [0.22959959592428564, 0.9104173736002714]



### The ANN model did pretty well.  There is some duplicate information in the dataset.  Let's see if removing that improves the performance of the ANN.

### Try ANN again with first 64 of 128 time values per row


```python
x_test = np.dstack((test1.iloc[:,0:64],test2.iloc[:,0:64],test3.iloc[:,0:64],test4.iloc[:,0:64],test5.iloc[:,0:64],test6.iloc[:,0:64],test7.iloc[:,0:64],test8.iloc[:,0:64],test9.iloc[:,0:64]))
x_train = np.dstack((train1.iloc[:,0:64],train2.iloc[:,0:64],train3.iloc[:,0:64],train4.iloc[:,0:64],train5.iloc[:,0:64],train6.iloc[:,0:64],train7.iloc[:,0:64],train8.iloc[:,0:64],train9.iloc[:,0:64]))
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape([7352,576])
x_test = x_test.reshape([2947,576])

```

    (7352, 64, 9)
    (2947, 64, 9)
    


```python
model = models.Sequential()
model.add(Dense(128, kernel_initializer="glorot_normal", input_shape=(9*64,)))
model.add(LeakyReLU())
model.add(Dropout(.15))
model.add(BatchNormalization())
# model.add(GaussianNoise(1))
model.add(Dense(64))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dense(6, activation='softmax'))
```


```python
# Let's use a different optimizer this time
noisy = add_gradient_noise(RMSprop)
model.compile(optimizer="Adamax",
# model.compile(optimizer=noisy(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```


```python
callbacks = [EarlyStopping(monitor='val_loss', patience=16),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
```


```python
history = model.fit(x_train, y_train, epochs=250, batch_size=16, validation_data = (x_test, y_test), callbacks=callbacks)
from keras.models import load_model
# model = load_model('best_model.h5', custom_objects={"NoisyRMSprop":noisy()})
model = load_model('best_model.h5')
model.evaluate(x_test, y_test)
```

    2947/2947 [==============================] - 0s 141us/step
    




    [0.24741540653827027, 0.9124533423820834]



### There did not seem to be much difference between the 2 ANN models, but how will an LSTM perform?


```python
x_test = np.dstack((test1,test2,test3,test4,test5,test6,test7,test8,test9))
x_train = np.dstack((train1,train2,train3,train4,train5,train6,train7,train8,train9))
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (7352, 128, 9)
    (2947, 128, 9)
    (7352, 6)
    (2947, 6)
    


```python
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import MobileNet
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Flatten, Dense, LeakyReLU, Dropout, GaussianNoise, LSTM
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

```


```python
# Let's use a different optimizer this time
noisy = add_gradient_noise(Adam)
model.compile(optimizer="adam",
# model.compile(optimizer=noisy(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```


```python
callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
```


```python
# history = model.fit(x_train, y_train, epochs=250, batch_size=32, validation_data = (x_test, y_test), callbacks=callbacks)
# from keras.models import load_model
# model = load_model('best_model.h5', custom_objects={"NoisyAdam":noisy()})
model = load_model('best_model.h5')
model.evaluate(x_test, y_test)
```

    2947/2947 [==============================] - 4s 1ms/step
    




    [0.2557287041376691, 0.9097387173396675]



### Conclusion:

### A LSTM does about the same as an ANN on this dataset.  XGBoost and SVM were the best models.
