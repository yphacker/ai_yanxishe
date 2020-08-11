# coding=utf-8
# author=yphacker

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv('../data/train.csv')
data.drop('id', replace=True)
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar',
                'rest_ecg', 'max_heart_rate',
                'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

x = data.corr()
pd.DataFrame(x['target']).sort_values(by='target', ascending=False).style.background_gradient(cmap='copper')

data.sex = data.sex.map({0: 'female', 1: 'male'})

data.chest_pain_type = data.chest_pain_type.map(
    {1: 'angina pectoris', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'SMI', 0: 'absent'})

data.fasting_blood_sugar = data.fasting_blood_sugar.map({0: 'lower than 120mg/ml', 1: 'greater than 120mg/ml'})

data.exercise_angina = data.exercise_angina.map({0: 'no', 1: 'yes'})

data.st_slope = data.st_slope.map({1: 'upsloping', 2: 'horizontal', 3: 'downsloping', 0: 'absent'})

data.thalassemia = data.thalassemia.map({1: 'normal', 2: 'fixed defect', 3: 'reversable defect', 0: 'absent'})

data.head()

X = data.iloc[:, 0:13]

Y = data.iloc[:, -1]

num_columns = ['age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate', 'st_depression']

ss = StandardScaler()

X[num_columns] = ss.fit_transform(X[num_columns])

categorical_columns = X.select_dtypes(include=['object']).head().columns

for column in categorical_columns:
    dummies = pd.get_dummies(X[column], drop_first=True)
    X[dummies.columns] = dummies
    X.drop(column, axis=1, inplace=True)

temp = X.copy()
temp['target'] = Y

d = temp.corr()
pd.DataFrame(d['target']).sort_values(by='target', ascending=False).style.background_gradient(cmap='copper')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X-Train:", X_train.shape)
print("X-Test:", X_test.shape)
print("Y-Train:", y_train.shape)
print("Y-Test:", y_test.shape)


def show_metrics(model):
    fig = plt.figure(figsize=(25, 10))

    # Confusion matrix
    fig.add_subplot(121)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"size": 16}, fmt='g')

    # ROC Curve
    fig.add_subplot(122)

    auc_roc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    plt.plot(fpr, tpr, color='darkorange', lw=2, marker='o', label='Trained Model (area = {0:0.3f})'.format(auc_roc))
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--', label='No Skill (area = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# creating our model instance
log_reg = LogisticRegression()

# fitting the model
log_reg.fit(X_train, y_train)

# predicting the target vectors
y_pred = log_reg.predict(X_test)

show_metrics(log_reg)

accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using Logistic Regression is: {np.round(accuracy, 3) * 100.0}%")

print(classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier

# creating a list of K's for performing KNN
my_list = list(range(0, 30))

# filtering out only the odd K values
neighbors = list(filter(lambda x: x % 2 != 0, my_list))

# list to hold the cv scores
cv_scores = []

# perform 10-fold cross validation with default weights
for k in neighbors:
    Knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    scores = cross_val_score(Knn, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
    cv_scores.append(scores.mean())

# finding the optimal k
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print("The optimal K value is with default weight parameter: ", optimal_k)

# plotting accuracy vs K
plt.plot(neighbors, cv_scores)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K Plot for normal ")
plt.grid()
plt.show()

print("Accuracy scores for each K value is : ", np.round(cv_scores, 3))

from sklearn.metrics import accuracy_score

# create instance of classifier
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k, algorithm='kd_tree',
                                   n_jobs=-1)

# fit the model
knn_optimal.fit(X_train, y_train)

# predict on test vector
y_pred = knn_optimal.predict(X_test)

# evaluate accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"The accuracy on test set using KNN for optimal K = {optimal_k} is {np.round(accuracy, 3)}%")

show_metrics(knn_optimal)

# Creating an instance of the classifier
svm = SVC()

# training on train data
svm.fit(X_train, y_train)

# predicting on test data
y_pred = svm.predict(X_test)

# let's look at our accuracy
accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using SVC is: {np.round(accuracy, 3) * 100.0}%")

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_

# Creating an instance for the classifier
rf_best = RandomForestClassifier(**rf_random.best_params_)

# fitting the model
rf_best.fit(X_train, y_train)

# predict the labels
y_pred = rf_best.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using RandomForest is: {np.round(accuracy, 3) * 100.0}%")

# creating a list of our models
ensembles = [log_reg, knn_optimal, rf_best, svm]

# Train each of the model
for estimator in ensembles:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

scores = [estimator.score(X_test, y_test) for estimator in ensembles]

scores

named_estimators = [
    ("log_reg", log_reg),
    ('random_forest', rf_best),
    ('svm', svm),
    ('knn', knn_optimal),
]

voting_clf = VotingClassifier(named_estimators)

voting_clf.fit(X_train, y_train)

# Let's look at our accuracy
acc = voting_clf.score(X_test, y_test)

print(f"The accuracy on test set using voting classifier is {np.round(acc, 3) * 100}%")

# to generate permutations of length three
perm = permutations(named_estimators, 3)

# to store the acc and classifiers
best_perm = []

# to store best classifier
best = []

# Traverse through the obtained permutations
for i in list(perm):
    # fit the classifier
    voting_clf = VotingClassifier(i)
    voting_clf.fit(X_train, y_train)

    # obtain accracy score and append it to the list
    acc = voting_clf.score(X_test, y_test)
    best_perm.append([acc, voting_clf])

# find out the maximum accuracy
maximum = max(best_perm, key=lambda x: x[0])

# there can be multiple permutations for which we get
# best score so find all of them and append to best
for i in range(len(best_perm)):
    if maximum[0] == best_perm[i][0]:
        best.append(best_perm[i][1])

acc_scores = []

for i in range(len(best)):
    voting_clf = best[i]

    # fit the classifier
    voting_clf.fit(X_train, y_train)

    # Let's look at our accuracy
    acc_scores.append(voting_clf.score(X_test, y_test))

print(f"The accuracy on test set using voting classifier is {np.round(max(acc_scores), 4) * 100}%")

# voting_clf.predict()
