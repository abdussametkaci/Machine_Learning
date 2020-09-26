# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:04:40 2019

@author: Abdussamet
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#veri Yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]#eğitim seviyeleri
y = veriler.iloc[:,2:]#maaşlar
X = x.values#x dataframe'in değerlerini alır
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X, Y)#X'ten Y'yi öğren (train ettiğimiz kısım), X i kullanarak Y yi öğren

plt.scatter(X, Y, color = "red")#X ileY yi çizer
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)#2. dereceden polinom
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")#ploynomial verileri linear'e çevirir ve çizer
plt.show()

poly_reg = PolynomialFeatures(degree = 4)#4. dereceden polinom
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.show()

#tahminler 
#NOT= predict fonksiyonu 2.dereceden dizi alır parametreolarak
print(lin_reg.predict([[11]]))#11.eğitim seviyesindekinin maaşını tahmin et
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))#11.eğitim seviyesindekinin maaşını tahmin et(polynomial regresyona göre)
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
y_olcekli = sc1.fit_transform(Y)

#Support Vektor Regression
from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")#kernel çizim şekli
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color = "red")#nokta
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color = "blue")#çizgi

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))