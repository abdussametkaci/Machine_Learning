# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:06:55 2019

@author: Abdussamet
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:, 3:].values

#K-Means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = "k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)#küme merkezlerini bastık

sonuclar = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 123)#her dönüşte random değişmemesi için bir başlangıç belirledik
    #random_satate herhangi bir değer olabilir, önemli olan her döngünün aynı random ile başlmaması
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)#wcss değerleini listeye atar

plt.plot(range(1, 11), sonuclar)#1 den 10 a kadarki değerleri alır
plt.show()

kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0, 0], X[Y_tahmin==0, 1], s=100, color="red")
plt.scatter(X[Y_tahmin==1, 0], X[Y_tahmin==1, 1], s=100, color="blue")
plt.scatter(X[Y_tahmin==2, 0], X[Y_tahmin==2, 1], s=100, color="green")
plt.scatter(X[Y_tahmin==3, 0], X[Y_tahmin==3, 1], s=100, color="yellow")
plt.title("KMeans")
plt.show()

#HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters= 4, affinity = "euclidean", linkage= "ward")
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0, 0], X[Y_tahmin==0, 1], s=100, color="red")
plt.scatter(X[Y_tahmin==1, 0], X[Y_tahmin==1, 1], s=100, color="blue")
plt.scatter(X[Y_tahmin==2, 0], X[Y_tahmin==2, 1], s=100, color="green")
plt.scatter(X[Y_tahmin==3, 0], X[Y_tahmin==3, 1], s=100, color="yellow")
plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()



