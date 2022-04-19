# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:44:11 2022

@author: Lilian FAVRE GARCIA
         Andrianihary Tsiory RAZAFINDRAMISA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# Pour encoder des données catégorielles
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Dataset:
    def __init__(self, path, display, selected_data=1, train_split=-1):
        """
        Initiate the dataset by loading the files given a path.

        Parameters
        ----------
        path : string
            The path to the "Data" folder of this project.
        display : boolean
            To display or not the histograms of the selected values.
        selected_data : float [0,1], optional
            The maximum proportion of data allowed in a 10% range whithin the
            total range values of the data. The default is 1.
        train_split : int [3,10], optional
            How much subsets of the train dataset we create in order to make a
            kcross validation for the train/test datasets.
            For example, if =3, we split the dataset Train in 3 subsets and
            one of them is going to be the Test dataset, then another one, and
            finally the last one. Can't be less than 3 because otherwise the
            Test dataset is going to be 50% of the total Train dataset. The
            higher is the value, the longer is the computation. The default is
            -1, wich is the value meaning that we don't make the kcross
            validation for the Train subset an we select 20% of it to be the
            Test.

        Returns
        -------
        None.
        """
        self.images = []
        self.split = train_split
        self.train = pd.read_csv(str(path + '/train.csv'))
        self.unknownData = pd.read_csv(str(path + '/test.csv'))    
        
        # preprocessing
        self.preprocess()
        to_delete = self.selectData(display, selected_data)
        self.feature_selection(to_delete)

        # because train only has the specie's name and we need a verify, we must split the train dataset
        # and to get a diversified dataset at each program run, we randomly shuffle it
        self.train = self.train.sample(frac=1)

        if train_split == -1:
            # We do not make the kcross validation for the data train/test
            # split
            self.Kcross = False
            # and we take the first 80% of the df to create the train and the
            # last 20% to create the test dataset
            self.test = self.train.iloc[int(self.train.shape[0]*0.8):]
            self.train = self.train.iloc[:int(self.train.shape[0]*0.8)]
        else:
            # We want to try different configurations of train/test splits
            self.Kcross = True
            self.cells = []

            for i in range(train_split):
                self.cells.append(self.train.iloc[int(self.train.shape[0]*i/train_split):int(self.train.shape[0]*(i+1)/train_split)])
            
            # We reset the datasets because they are going to be well initialized after
            self.train = pd.DataFrame()
            self.test = pd.DataFrame()

    def preprocess(self):
        """
        Calls the functions of data processing

        Returns
        -------
        None.
        """
        for (tr_columnName, tr_columnData) in self.train.iteritems():
            if (not tr_columnName == 'id') and (not tr_columnName == 'species'):  # TODO : on peux optimiser ?
                self.center_reduce(tr_columnName, tr_columnData)
                self.normalize(tr_columnName, tr_columnData)
                # useless
                # self.troncate(tr_columnName)

    def normalize(self, colName, colData):
        """
        Normalize the data in the given column name.

        Parameters
        ----------
        colName : string
            Name of the column to be normalized.
        colData : string
            Data of the column.

        Returns
        -------
        None.
        """
        _min = 0
        _max = 0

        tr_min = np.min(colData)
        tr_max = np.max(colData)
        te_min = np.min(self.unknownData.loc[:,colName])
        te_max = np.max(self.unknownData.loc[:,colName])
        
        if tr_min < te_min:
            _min = tr_min
        else:
            _min = te_min

        if tr_max > te_max:
            _max = tr_max
        else:
            _max = te_max

        for i in range(0, len(colData)):
            self.train.at[i,colName] = (self.train.at[i,colName] - _min) / (_max - _min)

            # the test has less values than the train dataset
            if i < len(self.unknownData.loc[:,colName]):
                self.unknownData.at[i,colName] = (self.unknownData.at[i,colName] - _min) / (_max - _min)
                
    def center_reduce(self, colName, colData):
        """
        Center and reduce the data in the given column name.

        Parameters
        ----------
        colName : string
            Name of the column to be normalized.
        colData : string
            Data of the column.


        Returns
        -------
        None.
        """
        mean = np.mean(colData)
        std = np.std(colData)
        
        for i in range (0,len(colData)):
            self.train.at[i,colName] = (self.train.at[i,colName] - mean) / std
            
            # the unknownData has less values than the train dataset
            if i < len(self.unknownData.loc[:,colName]):
                self.unknownData.at[i,colName] = (self.unknownData.at[i,colName] - mean) / std
        
    def troncate(self, colName):     
        """
        Troncate the data to reduce its precision.

        Parameters
        ----------
        colName : string
            Name of the column to be normalized.

        Returns
        -------
        None.
        """
        # TODO : mettre le 5 en valeur saisissable par l'utilisateur si cette fontion est pertinente
        self.train[colName] = self.train[colName].round(5)
        self.unknownData[colName] = self.unknownData[colName].round(5)

    def feature_selection(self, to_delete):
        """
        Deleted the data selected by the user.

        Parameters
        ----------
        to_delete : list[string]
            List of the columns names to delete.

        Returns
        -------
        None.
        """
        if not to_delete:  # lists are considered as bool if empty
            pass

        self.train.drop(columns=to_delete, axis=1, inplace=True)
        self.unknownData.drop(columns=to_delete, axis=1, inplace=True)

    def selectData(self, display, tolerance):
        """
        Select the data chosen by the user's value. Can plot the histograms of
        each caracteristic.

        Parameters
        ----------
        display : boolean
            To display or not the histograms of the selected values.
        tolerance : float [0,1]
            The maximum proportion of data allowed in a 10% range whithin the
            total range values of the data.

        Returns
        -------
        to_delete : list[string]
            List of string containing the names of the columns to be deleted.
        """
        to_delete = []
        if display:
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
            ignore = 2  # to ignore the 2 first columns (id and species)
            for (columnName, columnData) in self.train.iteritems():
                if ignore == 0:
                    n, _ = np.histogram(columnData.tolist())

                    if (np.max(n) >= np.sum(n) * tolerance):
                        to_delete.append(columnName)
                else:
                    ignore -= 1

        return to_delete

    def split_data(self, k):
        lst = []
        for ki in range(self.split):
            if (ki != k):
                lst.append(self.cells[ki][:])

        self.train = pd.concat(lst)
        self.test = self.cells[k]

    def get_Species(self):
        self.species = []

        for spe in self.train['species']:
            if spe not in self.species:
                self.species.append(spe)

        return self.species.sort()

    def xTrain(self):
        X = np.ndarray(shape=[2, self.train.shape[1]])
        X = self.train.loc[:, (self.train.columns != 'id') & (self.train.columns != 'species')]
        return X.to_numpy()

    def yTrain(self):
        t = np.ndarray(shape=[2, self.train.shape[1]])
        t = self.train.loc[:, ['species']]
        return t.to_numpy()

    def xTest(self):
        X = np.ndarray(shape=[2, self.test.shape[1]])
        X = self.test.loc[:, (self.test.columns != 'id') & (self.test.columns != 'species')]
        return X.to_numpy()

    def yTest(self):
        t = np.ndarray(shape=[2, self.test.shape[1]])
        t = self.test.loc[:, ['species']]
        return t.to_numpy()

    def xUnknownData(self):
        X = np.ndarray(shape=[2,self.unknownData.shape[1]]) 
        X = self.unknownData.loc[:,(self.unknownData.columns != 'id')] 
        return X.to_numpy() 