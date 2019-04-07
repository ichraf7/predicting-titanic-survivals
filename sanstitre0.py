#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:13:25 2019

@author: ichraf
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt
# data visualization
import seaborn as sns
data = pd.read_csv('titanic_data.csv')

#age and gender
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = data[data['Sex']=='female']
men = data[data['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde =False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde =False)
ax.legend()
ax.set_title('male')

per_survived_men=men[men['Survived']==1].shape[0]/men.shape[0]
per_died_men=men[men['Survived']==0].shape[0]/men.shape[0]

#Pclass and embarked port

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(10, 4))

C = data[data['Embarked']=='C']
S = data[data['Embarked']=='S']
Q = data[data['Embarked']=='Q']

ax = sns.distplot(S[S['Survived']==1].Pclass.dropna(), bins=3, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(S[S['Survived']==0].Pclass.dropna(), bins=3, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('S embarked port')

ax = sns.distplot(C[C['Survived']==1].Pclass.dropna(), bins=3, label = survived, ax = axes[1], kde =False)
ax = sns.distplot(C[C['Survived']==0].Pclass.dropna(), bins=3, label = not_survived, ax = axes[1], kde =False)
ax.legend()
ax.set_title('C embarked port')


ax = sns.distplot(Q[Q['Survived']==1].Pclass.dropna(), bins=3, label = survived, ax = axes[2], kde =False)
ax = sns.distplot(Q[Q['Survived']==0].Pclass.dropna(), bins=3, label = not_survived, ax = axes[2], kde =False)
ax.legend()
ax.set_title('Q embarked port')


#percent of survival per class
per_survived_port_S=S[S['Survived']==1].shape[0]/S.shape[0]
per_survived_port_C=C[C['Survived']==1].shape[0]/C.shape[0]
per_survived_port_Q=Q[Q['Survived']==1].shape[0]/Q.shape[0]

#class and survive
sns.barplot(x='Pclass', y='Survived', data=data)


ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde =False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde =False)
ax.legend()
ax.set_title('male')

per_survived_men=men[men['Survived']==1].shape[0]/men.shape[0]
per_died_men=men[men['Survived']==0].shape[0]/men.shape[0]




ax = sns.distplot(class1[class1['Survived']==1].Pclass.dropna(), bins=3, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(S[S['Survived']==0].Pclass.dropna(), bins=3, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('S embarked port')


#SibSp and Parch:
data['relatives']=data['SibSp']+data['Parch']
data=data.drop(['SibSp'],axis=1)
data=data.drop(['Parch'],axis=1)
sns.barplot(x='relatives', y='Survived', data=data)
sns.barplot(x='SibSp', y='Survived', data=data)
sns.barplot(x='Parch', y='Survived', data=data)

data=data.drop(['Name'],axis=1)
data=data.drop(['PassengerId'],axis=1)
data=data.drop(['Cabin'],axis=1)
data=data.drop(['Ticket'],axis=1)


X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
#DATA PREPROCESSING : AGE
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X[:,2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
#DATA PREPROCESSING : EMBARKED
X[61,5]='S'
X[829 ,5]='S'
data['Age']=X[:,2:3]
output=data['Survived']
features_raw =data.drop('Survived' ,axis=1)

#divide age into categories
for index , row  in data.iterrows():
       if (row['Age']>11.0) & (row['Age']<=18.0):
           X[index ,2]=1
       elif row['Age']<=11.0:
           X[index ,2]=0
       elif (row['Age']>18) & (row['Age']<=22):
           X[index ,2]=2
       elif (row['Age']>22) & (row['Age']<=30):
           X[index ,2]=3
       elif (row['Age']>30)&(row['Age']<=40):
           X[index ,2]=4
       elif (row['Age']>40)&(row['Age']<=50):
           X[index ,2]=5
       elif (row['Age']>50)&(row['Age']<=55):
           X[index ,2]=6
       elif (row['Age']>55):
           X[index ,2]=7 
data['Age']=X[:,2:3]          

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data['Fare'] = sc_X.fit_transform(data['Fare'])  

##numerical data
final_data=data['Fare']
#####transform categorical  data to numerical
categorical_features = ['Age', 'Sex','Embarked','Pclass']
# Get dummies for each categorical feature and concatenate it to the features dataframe
for feature in categorical_features:
   final_data= pd.concat([final_data, pd.get_dummies(data[feature], prefix = feature)], axis=1)
###end of data preprocessing
   
# Import train_test_split
from sklearn.cross_validation import train_test_split
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(final_data,output, test_size = 0.3, random_state = 0)   

# TODO: Import the three supervised learning models from sklearn
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

#Stochastic Gradient Descent (SGD)
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)   
   
#random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest_score=random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred_log = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
   
#kNN 
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train) 
Y_pred_KNN = knn.predict(X_test) 
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

#gaussian
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train) 
Y_pred_gauss = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)       

#perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred_percep = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#SVM
linear_svc = SVC(random_state=1990 ,kernel='sigmoid')
linear_svc.fit(X_train, Y_train)

Y_pred_SVM = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

#tree decision
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred_tree = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# GradientBoostingClassifier
gbs= GradientBoostingClassifier(random_state=1990)
gbs.fit(X_train,Y_train)
Y_pre_gbs=gbs.predict(X_test)

acc_decision_tree = round(gbs.score(X_train, Y_train) * 100, 2)

from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

#confusion metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

## precision and recall
from sklearn.metrics import precision_score, recall_score
print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
### f1 score 
from sklearn.metrics import f1_score
f1_score(Y_train, predictions)


#confusion metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions1 = cross_val_predict(random_forest, X_train, Y_test, cv=10)
confusion_matrix(Y_test, predictions)

## precision and recall
from sklearn.metrics import precision_score, recall_score
print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
### f1 score 
from sklearn.metrics import f1_score
f1_score(Y_train, predictions)


from sklearn.decomposition import PCA

features_reduced = PCA(n_components=2).fit_transform(final_data)

# Plot negative samples
mask_neg = np.array(survived==0, dtype=bool)
plt.scatter(features_reduced[mask_neg, 0], features_reduced[mask_neg, 1], color = 'pink',
            marker = 'o', edgecolors = 'black')


mask_pos = np.array(survived==1, dtype=bool)
plt.scatter(features_reduced[mask_pos,0], features_reduced[mask_pos,1], color = 'green',
            marker = '^', edgecolors = 'black')


plt.show()