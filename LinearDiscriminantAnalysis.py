# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:54:11 2022

@author: Tsiory
"""

# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


class LinearDiscriminantAnalysis():
    def __init__(self, dataHandler):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main

        Returns
        -------
        None.

        """
        self.LinearDiscriminant = None
        self.dh = dataHandler

        # self.proportion = proportion

        self.err_train = []
        # self.err_valid = []

    def hyperparameters_search(self, xData, yData):
        """
        Search for the best hyperparameters

        Parameters
        ----------
        xData : numpy.ndarray
            The data to predict
        yData : numpy.ndarray
            The labels

        Returns
        -------
        None.

        """
        self.LinearDiscriminant = LDA()
        c_v = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid = dict()
        grid['solver'] = ['svd', 'lsqr', 'eigen']
        grid['shrinkage'] = list(np.arange(0, 1, 0.01))
        grid['shrinkage'].append(None)
        grid['shrinkage'].append('auto')
        grid['tol'] = [0.0001, 0.001, 0.01]
        GSCV = GridSearchCV(self.LinearDiscriminant, grid, scoring='accuracy', cv=c_v, n_jobs=-1, verbose=5)
        results = GSCV.fit(xData, yData.ravel())
        print("Accuracy: %0.2f (+/- %0.2f)" % (results.best_score_, results.cv_results_['std'][0]))
        print("Best parameters set found on development set:", results.best_params_)
        self.err_train.append(self.score(xData, yData.ravel()))

    def score(self, xData, yData):
        """
        Compute the score of the model

        Parameters
        ----------
        xData : numpy.ndarray
            The data to predict
        yData : numpy.ndarray
            The labels

        Returns
        -------
        float
            The score of the model

        """
        return self.LinearDiscriminant.score(xData, yData)

    def run(self):
        """
        Run the model

        Returns
        -------
        List[string]
            Every class or label deduced from the entry dataset with the
            trained model.

        """
        return self.LinearDiscriminant.predict(self.dh.xUnknownData())
