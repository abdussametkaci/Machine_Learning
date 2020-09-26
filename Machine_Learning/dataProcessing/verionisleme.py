# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
veriler = pd.read_csv("veriler.csv")#csv dosyası okur
print(veriler)

#veri ön işleme
boy = veriler[["boy"]]
print(boy)

boykilo = veriler[["boy", "kilo"]]
print(boykilo)
