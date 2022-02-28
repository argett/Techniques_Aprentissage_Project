# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:44:11 2022

@author: Lilian FAVRE GARCIA
         Andrianihary Tsiory RAZAFINDRAMISA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import PercentFormatter

# Pour traiter les données manquantes
from sklearn.impute import SimpleImputer

# Pour encoder des données catégorielles
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

class Dataset:
    def __init__(self, path, selected_data):
        self.images = []
        self.train = pd.read_csv(str(path + 'train.csv'))
        self.test = pd.read_csv(str(path + 'test.csv'))
        
        for i in range(1,1585):
            self.images.append(mpimg.imread(str("Data/images/" + str(i) + ".jpg")))        
        
        # preprocessing
        self.remove_unselected_data(selected_data)
        self.normalize()
        print("stop")
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

    def normalize(self):
        for (tr_columnName, tr_columnData) in self.train.iteritems():
            if (not tr_columnName == 'id') and (not tr_columnName == 'species'): # TODO : on peux optimiser ?
                _min = 0
                _max = 0
                tr_min = np.min(tr_columnData)
                tr_max = np.max(tr_columnData)
                te_min = np.min(self.test.loc[:,tr_columnName])
                te_max = np.max(self.test.loc[:,tr_columnName])
                
                if tr_min < te_min:
                    _min = tr_min
                else:
                    _min = te_min
                    
                if tr_max > te_max:
                    _max = tr_max
                else:
                    _max = te_max   
                
                for i in range (0,len(tr_columnData)):
                    self.train.at[i,tr_columnName] = (self.train.at[i,tr_columnName] - _min) / (_max - _min)
                    
                    # the test has less values than the train dataset
                    if i < len(self.test.loc[:,tr_columnName]):
                        self.test.at[i,tr_columnName] = (self.test.at[i,tr_columnName] - _min) / (_max - _min)

    def plot_caracteristics(self):
        i = 0
        plt.subplots(figsize=(12, 12))
        for (columnName, columnData) in self.train.iteritems():
            if "margin" in columnName:
                i += 1
                plt.subplot(8, 8, i) 
                plt.hist(columnData)
                plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(columnData)))
                plt.title(columnName)
            elif "shape" in columnName:
                i += 1
                plt.subplot(8, 8, i) 
                plt.hist(columnData)
                plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(columnData)))
                plt.title(columnName)
            elif "texture" in columnName:
                i += 1
                plt.subplot(8, 8, i) 
                plt.hist(columnData)
                plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(columnData)))
                plt.title(columnName)
            
            if i == 64:
                plt.tight_layout()
                plt.show()
                plt.subplots(figsize=(12, 12))
                i = 0
            
    def remove_unselected_data(self, user_selection):
        if user_selection == 's':
            return
        
        # automatically it is mixed or hard so we remove mixed first
        self.train.drop(columns=['margin16', 'margin23', 'margin52', 'texture15', 'texture21', 'texture36', 'texture56', 'texture60', 'texture61'], axis=1, inplace=True)
        self.test.drop(columns=['margin16', 'margin23', 'margin52', 'texture15', 'texture21', 'texture36', 'texture56', 'texture60', 'texture61'], axis=1, inplace=True)
        
        if user_selection == 'h':
            self.train.drop(columns=['margin8', 'margin27', 'margin32', 'margin41', 'margin64','texture1', 'texture12', 'texture32', 'texture33', 'texture41', 'texture51'], axis=1, inplace=True)
            self.test.drop(columns=['margin8', 'margin27', 'margin32', 'margin41', 'margin64','texture1', 'texture12', 'texture32', 'texture33', 'texture41', 'texture51'], axis=1, inplace=True)

    def get_Species(self):
        self.species = []
        
        for spe in self.train['species'] :
            if not spe in self.species:
                self.species.append(spe)
                
        return self.species.sort()
    
    
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