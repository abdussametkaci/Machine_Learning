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
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

#Karar Ağacı (Decision Tree)
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X, Y)#X ve Y arasındaki ilişkiyi öğren
Z = X + 0.5
K = X - 0.4
plt.scatter(X, Y, color= "red")
plt.plot(x, r_dt.predict(X), color = "blue")#her bir X için X i çiz,X için tahmin değerlerini kullanarak
plt.plot(x, r_dt.predict(Z), color = "green")
plt.plot(x, r_dt.predict(K), color = "yellow")
#bu karar ağacı, belli bir aralığa giren değerler için aynı değeri döndürür
plt.show()#show demezsek arkada çizerken diğer kodları da çalıştırır ve 
#çizim bittiğinde çizer. bu bazı sıkıntılara yol açabilir
#ayrıca eğer başka çizim kodu varsa aynı tabloya ekleyebilir
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)#10 tane decision tree çizileceğini belirttik
#veriyi 10 parçaya bölerek decision tree'leri oluşturur
rf_reg.fit(X, Y)
print(rf_reg.predict([[6.6]]))#random forest birden fazla decision tree ye göre karar verdiği için verideki değerlerin dışında değer döndürebilir

plt.scatter(X, Y, color = "red")
plt.plot(x, rf_reg.predict(X), color = "blue")
plt.plot(x, rf_reg.predict(Z), color = "green")
plt.plot(x, rf_reg.predict(K), color = "yellow")
plt.show()

