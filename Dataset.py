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
    def __init__(self, path, display, selected_data):
        self.images = []
        self.train = pd.read_csv(str(path + 'train.csv'))
        self.test = pd.read_csv(str(path + 'test.csv'))
        
        for i in range(1,1585):
            self.images.append(mpimg.imread(str("Data/images/" + str(i) + ".jpg")))        
        
        # preprocessing
        self.preprocess()
        to_delete = self.plot_caracteristics(display, selected_data)
        self.feature_selection(to_delete)
        
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

    def preprocess(self):
        for (tr_columnName, tr_columnData) in self.train.iteritems():
            if (not tr_columnName == 'id') and (not tr_columnName == 'species'): # TODO : on peux optimiser ?
                self.center_reduce(tr_columnName, tr_columnData)
                self.normalize(tr_columnName, tr_columnData)
                self.troncate(tr_columnName)

    def normalize(self, colName, colData):
        _min = 0
        _max = 0
        
        tr_min = np.min(colData)
        tr_max = np.max(colData)
        te_min = np.min(self.test.loc[:,colName])
        te_max = np.max(self.test.loc[:,colName])
        
        if tr_min < te_min:
            _min = tr_min
        else:
            _min = te_min
            
        if tr_max > te_max:
            _max = tr_max
        else:
            _max = te_max   
        
        for i in range (0,len(colData)):
            self.train.at[i,colName] = (self.train.at[i,colName] - _min) / (_max - _min)
            
            # the test has less values than the train dataset
            if i < len(self.test.loc[:,colName]):
                self.test.at[i,colName] = (self.test.at[i,colName] - _min) / (_max - _min)
                
    def center_reduce(self, colName, colData):
        mean = np.mean(colData)
        std = np.std(colData)
        
        for i in range (0,len(colData)):
            self.train.at[i,colName] = (self.train.at[i,colName] - mean) / std
            
            # the test has less values than the train dataset
            if i < len(self.test.loc[:,colName]):
                self.test.at[i,colName] = (self.test.at[i,colName] - mean) / std
        
    def troncate(self, colName):     
        self.test[colName] = self.test[colName].round(5)
        self.train[colName] = self.train[colName].round(5)
        
        
    def feature_selection(self, to_delete):
        if not to_delete: # lists are considered as bool if empty
            pass
        
        # automatically it is mixed or hard so we remove mixed first
        self.train.drop(columns=to_delete, axis=1, inplace=True)
        self.test.drop(columns=to_delete, axis=1, inplace=True)
        
    def plot_caracteristics(self, display, tolerance):
        to_delete = []
        if display :
            i = 0
            plt.subplots(figsize=(12, 12))
            for (columnName, columnData) in self.train.iteritems():
                if "margin" in columnName:
                    i += 1
                    plt.subplot(8, 8, i) 
                    n, _, _ = plt.hist(columnData)
                    
                    if (np.max(n) >= np.sum(n) * tolerance):
                        plt.hist(columnData, color='red')
                        to_delete.append(columnName)
                    else:
                        plt.hist(columnData, color='blue')
                    
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(columnData)))
                    plt.title(columnName)
                elif "shape" in columnName:
                    i += 1
                    plt.subplot(8, 8, i) 
                    n, _, _ = plt.hist(columnData)
                    
                    if (np.max(n) >= np.sum(n) * tolerance):
                        plt.hist(columnData, color='red')
                        to_delete.append(columnName)
                    else:
                        plt.hist(columnData, color='blue')
                        
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(columnData)))
                    plt.title(columnName)
                elif "texture" in columnName:
                    i += 1
                    plt.subplot(8, 8, i) 
                    n, _, _ = plt.hist(columnData)
                    
                    if (np.max(n) >= np.sum(n) * tolerance):
                        plt.hist(columnData, color='red')
                        to_delete.append(columnName)
                    else:
                        plt.hist(columnData, color='blue')
                        
                    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=len(columnData)))
                    plt.title(columnName)
                
                if i == 64:
                    plt.tight_layout()
                    plt.show()
                    plt.subplots(figsize=(12, 12))
                    i = 0
        else:
            for (columnName, columnData) in self.train.iteritems():
                n, _, _ = plt.hist(columnData)
                    
                if (np.max(n) >= np.sum(n) * tolerance):
                    to_delete.append(columnName)
        
        return to_delete
                        
            

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
        t = self.train.loc[:,['species']] 
        return t.to_numpy() 
    
    def xTest(self):
        X = np.ndarray(shape=[2,self.test.shape[1]]) 
        X = self.train.loc[:,(self.test.columns != 'id')] 
        return X.to_numpy() 
    
    def train_getCaracteristics_id(self, id_):
        return self.train.loc[self.train["id"] == id_]   #	.tolist()
    
    def train_getCaracteristics_row(self, row):
        return self.train.iloc[row]   #	.tolist()

    def test_getCaracteristics_id(self, id_):
        return self.test.loc[self.train["id"] == id_]   #	.tolist()
    
    def test_getCaracteristics_row(self, row):
        return self.test.iloc[row]   #	.tolist()