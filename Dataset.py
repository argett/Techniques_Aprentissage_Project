# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:44:11 2022

@author: Lilian FAVRE GARCIA
         Andrianihary Tsiory RAZAFINDRAMISA
"""

import numpy as np
import pandas as pd
import matplotlib.image as mpimg

# Pour traiter les données manquantes
from sklearn.impute import SimpleImputer

# Pour encoder des données catégorielles
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

class Dataset:
    def __init__(self, path):
        self.images = []
        self.train = pd.read_csv(str(path + 'train.csv'))
        self.test = pd.read_csv(str(path + 'test.csv'))
        
        for i in range(1,1585):
            self.images.append(mpimg.imread(str("Data/images/" + str(i) + ".jpg")))

    def train_getCaracteristics_id(self, id_):
        return self.train.loc[self.train["id"] == id_]   #	.tolist()
    
    def train_getCaracteristics_row(self, row):
        return self.train.iloc[row]   #	.tolist()

    def test_getCaracteristics_id(self, id_):
        return self.test.loc[self.train["id"] == id_]   #	.tolist()
    
    def test_getCaracteristics_row(self, row):
        return self.test.iloc[row]   #	.tolist()

class Preprocessing:
    def __init__(self, path):
        self.images = []
        self.train = pd.read_csv(str(path + 'train.csv'))
        self.test = pd.read_csv(str(path + 'test.csv'))
        
        for i in range(1,1585):
            self.images.append(mpimg.imread(str("Data/images/" + str(i) + ".jpg")))

    def handling_missing(self, df, i, j): 
        # i et j les indices des colonnes contenant des valeurs numériques
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = df.iloc[:, i:j]
        imputer.fit(data)
        data = imputer.transform(data)
        return data