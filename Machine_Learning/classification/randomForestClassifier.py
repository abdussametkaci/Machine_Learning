# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:37:25 2019

@author: Abdussamet
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:, 1:4].values #bağımsız değişkenler
y = veriler.iloc[:, 4:].values #bağımlı değişkenler
print(y)

#verilerin egitim ve test icin bolunmesi
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)#fit eğitme , transform o eğitimi uygula demek
X_test = sc.transform(x_test)

#linear regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)#tahminlerimizle verileirmizin doğruluk ve yanlışlık oranlarını bir matriste döndürür
print(cm)

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1, metric = "minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# SVM (support vektor machine)
from sklearn.svm import SVC
svc = SVC(kernel = "linear")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("SVC:")
print(cm)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
 
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("GNB")
print(cm)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy")

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("DTC")
print(cm)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("RFC")
print(cm)
