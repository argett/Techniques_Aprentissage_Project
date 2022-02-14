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
        
        # preprocessing
        #self.train = self.handling_missing(self.train, 2, self.train.shape[1])
        #self.test = self.handling_missing(self.test, 2, self.test.shape[1])
        
    
    def handling_missing(self, df, i, j): 
        """
        Regarde dans le dataframe donné s'il y a des NaN pour changer par 0

        Parameters
        ----------
        df : Dataframe
            Le dataframe que l'on souhaite corriger
        i : int
            Valeur de la colonne du début où commence la vérification
        j : int
            Valaur de la colonne de fin où fini la vérification

        Returns
        -------
        data : Dataframe
            Le Dataframe corrigé

        """
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        data = df.iloc[:, i:j]
        imputer.fit(data)
        data = imputer.transform(data)
        return data
    
    def xTrain(self): 
        X = np.ndarray(shape=[2,self.train.shape[1]]) 
        X = self.train.loc[:,(self.train.columns != 'id') & (self.train.columns != 'species')] 
        return X.to_numpy() 
     
    def yTrain(self): 
        t = np.ndarray(shape=[2,self.train.shape[1]]) 
        t = self.train.loc[:,['id','species']] 
        return t.to_numpy() 
    
    def train_getCaracteristics_id(self, id_):
        return self.train.loc[self.train["id"] == id_]   #	.tolist()
    
    def train_getCaracteristics_row(self, row):
        return self.train.iloc[row]   #	.tolist()

    def test_getCaracteristics_id(self, id_):
        return self.test.loc[self.train["id"] == id_]   #	.tolist()
    
    def test_getCaracteristics_row(self, row):
        return self.test.iloc[row]   #	.tolist()
