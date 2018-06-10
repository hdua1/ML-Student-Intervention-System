# Building a Student Intervention System

About project - Given student data, identify students who might need early intervention before they fail to graduate.

We have been provided with student data on which analysis needs to be done to determine whether student needs intervention or not. This is a classification problem that requires examples to be categorized into two or more classes that can in-turn be fed into the learning algorithm as training data.

## Exploring the data

Total number of students: 395

Number of features: 30

Number of students who passed: 265

Number of students who failed: 130

Graduation rate of the class: 67.09%

## Preparing the data

### Feature Columns

['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

Target column: passed

### Preprocessed Feature Columns

Processed feature columns (48 total features):
['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']


### Training & Test Data

Training set: 300 samples
Test set: 95 samples

## Training & Evaluation Models

I chose the following four supervised learning models available in scikit-learn for the student data:

    1) Support Vector Machines(SVM)

    2) Gaussian Naive Bayes (GaussianNB)

    3) Logistic Regression

    4) Random Forest

### Support Vector Machines
SVM's working can be explained with the help of maximal-margin classifier. Consider you have some input variables or columns in the given data, 30 in this case, then the maximal-margin classifier will form a 30-dimension space. A hyperplane is a line that splits the input variable space. A hyperplane is selected to separate or classify the input points by their class. The distance between the points and the separated line, called margin is chosen such that it separates the two classes by maximum margin, hence called maximal-margin classifier. The margin is calculated as the perpendicular distance from the line to the closest points only. These points are referred to as support vectors. So, here SVM will first learn the data and classify the existing data into two groups, called labels, i.e., YES - in case the student needs intervention or NO - no intervention.

Advantages: 

    1) By introducing the kernel, SVMs gain flexibility in the choice of the form of the threshold separating input points, 
    2) The input points need not be linear and even need not have the same functional form for all data, since its function is non-parametric and operates locally.
    3) The SVM is an effective tool in high-dimensional spaces, which is particularly applicable to document classification and sentiment analysis where the dimensionality can be extremely large (≥10^6).

Disadvantages:

    1) SVMs don't work well with large datasets as the time complexity of training them is of the order of O(N^3).
    2) In situations where the number of features for each object exceeds the number of training data samples, SVMs can perform poorly. This can be seen intuitively, as if the high-dimensional feature space is much larger than the samples, then there are less effective support vectors on which to support the optimal linear hyperplanes.
    3) The results are not good in case of overlapping classes or data containing lots of noise.
    
### SVM Model Measurements

| Training Set Size | Training Time (sec) | Prediction Time (sec) | Training F1 | Test F1 |
| ----------------- | ------------------- | --------------------- | ----------- | ------- |
|       100         |       0.0023        |        0.0011         |     0.8383  |  0.8050 |
|       200         |       0.0045        |        0.0033         |     0.8371  |  0.8341 |
|       300         |       0.0080        |        0.0054         |     0.8677  |  0.8082 |


### Gaussian Naîve Bayes (Gaussian NB)

Based on Bayes theorem (a theorem which provides a way to calculate the probability of a hypothesis given our prior knowledge) and Naive Bayes Classifier(a classification algorithm for binary or multi-class classification problems using class probabilities and conditional probabilities), the Gaussian NB algorithm calculates the mean and standard deviation for input values for each class to summarize the distribution, in addition to the probabilities of each class.

Advantages:

    1) Fairly simple method that involves some counts that involves small amount of training data. If the NB conditional independence assumption actually holds, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data.
    
    2) Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods.The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
    
Disadvantages:

    1) Despite the above advantages, the estimations provided by it are bad.
    
    2) If the number of dependent attributes or parameters are large, then performance is poor.
    
 ### GaussianNB Model Measurements

| Training Set Size | Training Time (sec) | Prediction Time (sec) | Training F1 | Test F1 |
| ----------------- | ------------------- | --------------------- | ----------- | ------- |
|       100         |       0.0041        |        0.0014         |     0.8163  |  0.8160 |
|       200         |       0.0021        |        0.0028         |     0.7839  |  0.7520 |
|       300         |       0.0015        |        0.0009         |     0.7781  |  0.7541 |


### Logistic Regression

Logistic regression models the probability of the default class. For example, if we are modeling people’s sex as male or female from their height, then the first class could be male and the logistic regression model could be written as the probability of male given a person’s height, or more formally: P(sex=male|height). The probability prediction must be transformed into a binary values (0 or 1) in order to actually make a probability prediction. Logistic regression is a linear method, but the predictions are transformed using the logistic function. The binary logistic model is used to estimate the probability of a binary response based on one or more predictor (or independent) variables (features).

Advantages:

    1) Unlike decision trees or SVMs, model updation can be done easily. If you want a probabilistic framework (e.g., to easily adjust classification thresholds, to say when you’re unsure, or to get confidence intervals) or if you expect to receive more training data in the future that you want to be able to quickly incorporate into your model, then this method is beneficial.
    2) Logistic regression will work better if there's a single decision boundary, not necessarily parallel to the axis.
    
Disadavantages:

    1) Logistic regression attempts to predict outcomes based on a set of independent variables, but if researchers include the wrong independent variables, the model will have little to no predictive value. 

    2) It requires that each data point be independent of all other data points. If observations are related to one another, then the model will tend to overweight the significance of those observations. 
    
### Logistic Regression Model Measurements

| Training Set Size | Training Time (sec) | Prediction Time (sec) | Training F1 | Test F1 |
| ----------------- | ------------------- | --------------------- | ----------- | ------- |
|       100         |       0.0027        |        0.0008         |     0.9231  |  0.7107 |
|       200         |       0.0033        |        0.0010         |     0.8500  |  0.7820 |
|       300         |       0.0048        |        0.0008         |     0.8233  |  0.8029 |

### Random Forest Classifier
Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction. With a few exceptions a random-forest classifier has all the hyperparameters of a decision-tree classifier and also all the hyperparameters of a bagging classifier, to control the ensemble itself. Instead of building a bagging-classifier and passing it into a decision-tree-classifier, you can just use the random-forest classifier class, which is more convenient and optimized for decision trees. 

Advantages:

    1) One big advantage of random forest is, that it can be used for both classification and regression problems, which form the majority of current machine learning systems.
    
    2) It is very easy to measure the relative importance of each feature on the prediction.
    
    3) It handles high dimensional spaces as well as large number of training examples really well.
    
    4) Random forest runtimes are quite fast, and they are able to deal with unbalanced and missing data
    
Disadvantages:

    1) Random forests tends to overestimate the low values and underestimate the high values. This is because the response from random forests in the case of regression is the average (mean) of all of the trees.
    
    2) When used for regression they cannot predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy.

### Random Forest Model Measurements 

| Training Set Size | Training Time (sec) | Prediction Time (sec) | Training F1 | Test F1 |
| ----------------- | ------------------- | --------------------- | ----------- | ------- |
|       100         |       0.0429        |        0.0024         |     0.9928  |  0.8000 |
|       200         |       0.0417        |        0.0021         |     0.9885  |  0.7259 |
|       300         |       0.0425        |        0.0038         |     0.9901  |  0.7778 |


## Choosing the best model

For dataset of size approximately 300 and after considering factors such as F1 scores for different sized datasets, over-fitting, time complexity of each model, **SVM** is the best and the most optimal model for this purpose. If the data size were very large, Random Forest Classifier might have outperformed SVM in terms of time complexity.

### Best model produced from ``GridSearchCV``
#### Best combination of parameters

    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.01, verbose=False)
    
### Model's final F1 score

`0.978102189781`
