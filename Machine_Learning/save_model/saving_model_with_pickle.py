# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:18:21 2019

@author: asus
"""

import pandas as pd

url = "http://www.bilkav.com/wp-content/uploads/2018/03/satislar.csv"
veriler = pd.read_csv(url)
veriler = veriler.values
X = veriler[:, 0:1]
Y = veriler[:, 1]

bolme = 0.33

from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = bolme)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
print(lr.predict(X_test))

import pickle

dosya = "model.kayit"
# oluşturduğumuz modeli bir doyaya kaydedeceğiz, bu sayede daha sonra tekrar tekrar kullanabiliriz
# makinenin öğrendikleri artık bir dosyada!!!
pickle.dump(lr, open(dosya, "wb"))

#kaydedilen model aşağıdaki gibi açılır
yuklenen = pickle.load(open(dosya, "rb"))#bu aslında lr objesidir
#kanıt
print(yuklenen.predict(X_test))