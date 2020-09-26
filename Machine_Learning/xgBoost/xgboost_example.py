# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:41:35 2019

@author: Abdussamet
"""

# kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Veri Yukleme
veriler = pd.read_csv('Churn_Modelling.csv')

# Veri Kümesi
X = veriler.iloc[:, 3:13].values
Y = veriler.iloc[:, 13].values

# Ö İşleme
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] =labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#verilerin egitim ve test icin bolunmesi
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
print(cm)