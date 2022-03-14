# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:48:20 2022

@author: Lilian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class randomForest():
    def __init__(self, dataHandler, nb_trees=50, max_depth=100, min_sample=2, random_state=10, max_features=1, criterion="gini", proportion=0.2):
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
        self.randomForest = None
        
        self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(dataHandler.xTrain(), dataHandler.yTrain(), train_size=proportion, random_state=0) 
        
        self.dh = dataHandler
        self.proportion = proportion
        
        self.trees = nb_trees
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.rdm_state = random_state
        self.min_sample = min_sample
    
    def recherche_hyperparametres(self, num_fold, nb_trees, maxDepth, random_state, max_features, min_sample, criterion):
        meilleur_err = np.inf
        meilleur_tree = None
        meilleur_depth = None
        meilleur_state = None
        meilleur_feature = None
        meilleur_sample = None
        
        for tree in tqdm(nb_trees):  # On teste plusieurs degrés du polynôme
            for depth in maxDepth:
                for state in random_state:
                    for feature in max_features:
                        for sample in min_sample:
                            sum_error = 0
                            
                            self.trees = tree
                            self.max_features = feature
                            self.max_depth = depth
                            self.rdm_state = state
                            self.min_sample = sample
                            self.criterion = criterion
                
                            for k in range(num_fold):  # K-fold validation
                                self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=k, shuffle=True)
                                sum_error += self.entrainement()                                    
                                
                            avg_err_locale = sum_error/(num_fold)  # On regarde la moyenne des erreurs sur le K-fold  
                            if(avg_err_locale < meilleur_err):
                                meilleur_err = avg_err_locale
                                meilleur_tree = tree
                                meilleur_depth = depth
                                meilleur_state = state
                                meilleur_feature = feature
                                meilleur_sample = sample
                                    
        self.trees = meilleur_tree
        self.max_features = meilleur_feature
        self.max_depth = meilleur_depth
        self.rdm_state = meilleur_state
        self.min_sample = meilleur_sample
        
        print("trees = " + str(meilleur_tree))
        print("max_features = " + str(meilleur_feature))
        print("meilleur_depth = " + str(meilleur_depth))
        print("meilleur_state = " + str(meilleur_state))
        print("meilleur_sample = " + str(meilleur_sample))
        
    def entrainement(self):
        self.randomForest = RandomForestClassifier(n_estimators=self.trees, max_depth=self.max_depth, min_samples_split=self.min_sample,criterion=self.criterion, max_features=self.max_features)
        # TODO : retourner le score ?
        self.randomForest.fit(self.X_learn, self.y_learn.ravel()) # on utilise toutes les données d'entrainement
        return self.randomForest.score(self.X_verify, self.y_verify)
    
    def run(self):
        print(self.randomForest.predict(self.dh.xTest()))
