# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:14:34 2019

@author: Abdussamet
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv("Restaurant_Reviews.csv")

import re
"""
yorum = re.sub("[^a-zA-z]", " ", yorumlar["Review"][0])
yorum = yorum.lower()
yorum = yorum.split()
"""
import nltk

stopwords = nltk.download("stopwords")

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

from nltk.corpus import stopwords

# Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    yorum = re.sub("[^a-zA-z]", " ", yorumlar["Review"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    derlem.append(yorum)
# Feature Extraction (Öznitelik Çıkarımı)
    # Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2000)#en fazla kullanılan 2000 kelimeyi aldaha fazlasını alma
X = cv.fit_transform(derlem).toarray()# bağımsız değişken
y = yorumlar.iloc[:, 1].values # bağımlı değişken

# makine öğrenmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)