# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:01:04 2019

@author: Abdussamet
"""

import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

veriler = pd.read_csv("eksikveriler.csv")
print(veriler)

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#bos verilerin değeri NaN, strateg verilerin ortalamasını almak, eksen ise y ekseni

yas = veriler.iloc[:,1:4].values#verilerin tüm satırlarında 1 ile 4. sütun arasındaki değerleri al
print(yas)
imputer = imputer.fit(yas[:,1:4])#startejimizi gerçekleştirmeyi sağlar
yas[:,1:4] = imputer.transform(yas[:,1:4])#en sonda boi olan veriler yerine uygulanan strateji yönünde veriler değiştirilir
print(yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)
#Nominal, Ordinal ->Numerical
le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:,0])#hem fit hem de transform fonksiyonları çalışır
#LabelEncoder'ın bu fonksiyonu kategorize edilmiş veilere sayısal bir değer atar

print(ulke)

ohe = OneHotEncoder(categorical_features = "all")#tum kategoriler için
ulke = ohe.fit_transform(ulke).toarray()#denk gelen ulkelere 1 diğerlerine 0 verir
print(ulke)
#ulke adlı numpy dizisini datafarme'e çevirerek bir veri oluşturcaz ve sonrasında farklı verilerle birleştireceğiz
#veri=ulke dizisi, index 22.satıra kadar, colon başlıkları dizi içinde belitilir
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(22), columns=["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns = ["cinsiyet"])
print(sonuc3)
#eğer aaxis=1 demezsek dataları alt alta birleştir ve sütuna karşılık gelmeyen bir bilgi varsa NaN yazılır
s=pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

#verileri test ve eğitim olarak bölme
#NOT=cross_validation yerine model_selection yazılmalı, kütüphane değiştirilmiş
from sklearn.model_selection import train_test_split
#bağımsız değişkenler s(ilk kısım), bağlı değişkenler sonuc3(ikinci kısım), test boyutu 1/3, rastegle veriler alacak yani karışık seçecek
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

#standartlaştırma
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)












