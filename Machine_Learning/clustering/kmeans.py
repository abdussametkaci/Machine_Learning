# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:30:33 2019

@author: Abdussamet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:, 3:].values

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