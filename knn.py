# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:53:11 2022

@author: Lilian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class knn():
    def __init__(self, dataHandler, k, proportion=0.2 ,manhattan=False):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        k : int
            The number of neighbour we want to take into account in the KNN algorithm
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The default is 0.2.
        manhattan : boolean, optional
            Select the Euclidean distance by default, select the Manhattan distance if True. The default is False.

        Returns
        -------
        None.

        """
        t = dataHandler.yTrain()
        x = dataHandler.xTrain()
        self.X_learn, self.X_verify, y_learn, y_verify = train_test_split(dataHandler.xTrain(), dataHandler.yTrain(), train_size=proportion, random_state=0)
        
        
        self.nb_neighbour = k
        self.accuracy = 0
        self.distance = 0 # euclidian distance
        self.distance_method = manhattan # euclidean distance by default
        
    def run(self):
        reseau_knn = KNeighborsClassifier(self.nb_neighbour)
        reseau_knn.fit()
