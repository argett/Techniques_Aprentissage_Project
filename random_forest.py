# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:48:20 2022

@author: Lilian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class randomForest():
    def __init__(self, dataHandler, nb_trees, criterion="gini", max_features=None, random_state=5, proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        nb_trees : int
            The number of trees in the forest, 10 < x < 1000
        criterion : {"gini", "entropy"}, optional
            The number of trees in the forest. Default = "Gini"
        max_features : int, optional
            The maximum depth of the tree. Default = the number of features
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. Default = 0.2.

        Returns
        -------
        None.

        """
        self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(dataHandler.xTrain(), dataHandler.yTrain(), train_size=proportion, random_state=0)
        
        
        self.trees = nb_trees
        self.criterion = criterion
        self.max_features = max_features
        self.rdm_state = random_state
        
    def run(self):
        random_forest = RandomForestClassifier(n_estimators=self.trees, criterion=self.criterion, max_features=self.max_features)
        random_forest.fit(self.X_learn, self.y_learn.ravel())
        print(random_forest.score(self.X_verify, self.y_verify))
