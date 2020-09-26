# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:27:34 2019

@author: Abdussamet
"""

import pandas as pd
from sklearn.preprocessing import Imputer
veriler = pd.read_csv("eksikveriler.csv")
print(veriler)

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#bos verilerin değeri NaN, strateg verilerin ortalamasını almak, eksen ise y ekseni

yas =veriler.iloc[:,1:4].values#verilerin tüm satırlarında 1 ile 4. sütun arasındaki değerleri al
print(yas)
imputer = imputer.fit(yas[:,1:4])#startejimizi gerçekleştirmeyi sağlar
yas[:,1:4] = imputer.transform(yas[:,1:4])#en sonda boi olan veriler yerine uygulanan strateji yönünde veriler değiştirilir
print(yas)
