
# coding: utf-8

# # Project: Building a Student Intervention System

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.

# In[1]:


# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"


# ### Implementation: Data Exploration
# Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
# - The total number of students, `n_students`.
# - The total number of features for each student, `n_features`.
# - The number of those students who passed, `n_passed`.
# - The number of those students who failed, `n_failed`.
# - The graduation rate of the class, `grad_rate`, in percent (%).
# 

# In[2]:


# TODO: Calculate number of students
n_students = len(student_data.index)

# TODO: Calculate number of features
n_features = len(student_data.columns) - 1

# TODO: Calculate passing students
n_passed = len(student_data[student_data.passed=="yes"])

# TODO: Calculate failing students
n_failed = len(student_data[student_data.passed=="no"])

# TODO: Calculate graduation rate
grad_rate = (n_passed*100*1.0)/n_students

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.

# In[3]:


feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


# ### Preprocess Feature Columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.

# In[6]:


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Implementation: Training and Testing Data Split
# So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
# - Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
#   - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
#   - Set a `random_state` for the function(s) you use, if provided.
#   - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

# In[8]:


from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
sss = StratifiedShuffleSplit(y_all,test_size=num_test, random_state=0)
for train_index, test_index in sss:
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data


# #### Training and Evaluating Models
# In this section, you will choose 4 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.
# 
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Setup
# Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
# - `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
# - `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
# - `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
#  - This function will report the F<sub>1</sub> score for both the training and testing data separately.

# In[9]:


import time

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    print "Training model {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    
    # Print the results
    print "Model training completed in {:.4f} seconds".format(end - start)
    
from sklearn import svm
clf = svm.SVC()

#call the classifier method to train the data using SVM
train_classifier(clf,X_train,y_train)
print clf


# In[10]:


from sklearn.metrics import f1_score
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    
    # Print and return results
    print "Predictions made in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

trained_f1_score = predict_labels(clf,X_train,y_train)
print "F1 score for training dataset: {}".format(trained_f1_score)


# In[11]:


#finding f1 score for test data
print "F1 score for test dataset: {}".format(predict_labels(clf,X_test,y_test))


# In[12]:


from sklearn.decomposition import PCA
import pylab as pl

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set of size {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    print "\n\n"
    
train_predict(clf,X_train[:100],y_train[:100],X_test, y_test)
train_predict(clf,X_train[:200],y_train[:200],X_test, y_test)
train_predict(clf,X_train[:300],y_train[:300],X_test, y_test)


# ### Implementation: Model Performance Metrics
# With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
#  - Use a `random_state` for each model you use, if provided.
#  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Create the different training set sizes to be used to train each model.
#  - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
# - Fit each model with each training set size and make predictions on the test set (9 in total).  
# **Note:** Three tables are provided after the following code cell which can be used to store your results.

# In[14]:


from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

clfA = GaussianNB()
train_predict(clfA,X_train[:100],y_train[:100],X_test, y_test)
train_predict(clfA,X_train[:200],y_train[:200],X_test, y_test)
train_predict(clfA,X_train[:300],y_train[:300],X_test, y_test)

clfB = linear_model.LogisticRegression(C=1e5)
train_predict(clfB,X_train[:100],y_train[:100],X_test, y_test)
train_predict(clfB,X_train[:200],y_train[:200],X_test, y_test)
train_predict(clfB,X_train[:300],y_train[:300],X_test, y_test)

clfC = RandomForestClassifier(n_estimators=10)
train_predict(clfC,X_train[:100],y_train[:100],X_test, y_test)
train_predict(clfC,X_train[:200],y_train[:200],X_test, y_test)
train_predict(clfC,X_train[:300],y_train[:300],X_test, y_test)


# ## Choosing the Best Model
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
# - Initialize the classifier you've chosen and store it in `clf`.
# - Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
#  - Set the `pos_label` parameter to the correct value!
# - Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.

# In[16]:


from sklearn import grid_search
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

clf = svm.SVC()
param_grid = [
  {'C': [1,10, 50, 100, 200, 250, 300, 350, 400, 500, 600],
    'kernel':['rbf','poly','sigmoid'],
    'gamma': [0.001,0.01,0.1,1,0.1,0.01,0.001,0.0001,0.00001],
     'tol':[0.01,0.001,0.0001,0.00001,0.0000001]
  }
 ]

f1_scorer = make_scorer(f1_score, pos_label="yes")
regressor = grid_search.GridSearchCV(clf, param_grid, cv=5,scoring=f1_scorer)
regressor.fit(X_train, y_train)
reg = regressor.best_estimator_
print reg
print "\n"
train_predict(reg, X_train, y_train,X_test,y_test)

