# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:40:24 2019

@author: Abdussamet
"""

import pandas as pd
import matplotlib.pyplot as plt
import random

veriler = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000 #10000 tıklama
d = 10 #10 ilan var
toplam = 0 #tum odullerin toplamı
secilenler = []
birler = [0] * d
sıfırlar = [0] * d
for n in range(1, N):
    ad = 0 #seçilen ilan
    max_th = 0
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sıfırlar[i] + 1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n, ad]
    if odul == 1:
        birler[ad] = birler[ad] + 1
    else:
        sıfırlar[ad] = sıfırlar[ad] + 1
    
    toplam = toplam + odul

print("Toplam Odul")
print(toplam)   

plt.hist(secilenler)
plt.show()    
