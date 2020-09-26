# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:01:04 2019

@author: Abdussamet
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv("satislar.csv")
print(veriler)

aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

#verilerin egitim ve test icin bolunmesi
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
"""
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
#model inşaası (linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)#teste bakaraj tahmin yapar

x_train = x_train.sort_index()#x_train içindeki verileri indexlerine göre sıralar
#eğer indexleri sıralamazsak çizeceğimiz grafikte rastgele sıradaki verilerin çizimi olur
#rasthele olmasının nedeni de makineyi eğitirken rastgele veri seçmemizdi
#verileri doğru sıraya sokarsak grafik doğru olur
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
