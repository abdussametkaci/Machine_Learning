# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:48:22 2019

@author: Abdussamet
"""
#NOT:UCB kütüphanesi yok
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000 #10000 tıklama
d = 10 #10 ilan var
#Ri(n)
oduller = [0] * d #her bir elemanı 0 olan 10 elemanlı dizi (ilk başta tum ilanların odulu 0)
#Ni(n)
tıklamalar = [0] * d #o ana kadarki tıklamalar
toplam = 0 #tum odullerin toplamı
secilenler = []
for n in range(1, N):
    ad = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0, d):
        if tıklamalar[i] > 0:
            ortalama = oduller[i] / tıklamalar[i]
            delta = math.sqrt(3/2*math.log(n)/tıklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:#max'tan buyuk ucb çıktı
            max_ucb = ucb
            ad = i
    
    secilenler.append(ad)
    tıklamalar[ad] = tıklamalar[ad] + 1
    odul = veriler.values[n, ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul

print("Toplam Odul")
print(toplam)   

plt.hist(secilenler)
plt.show()    
