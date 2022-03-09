# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:21:33 2022

@author: Lilian
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


class gradientBoosting():
    def __init__(self, dataHandler, lr=0.0001, estimators=100, min_sample=2, proportion=0.8):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        lr : float, optional
            The learning rate corresponding to the gradient descent
        estimators : int, optional
            The number of boosting stages to perform. A large number usually results in better performance
        min_sample : int, optional
            The minimum number of samples required to split an internal node
        proportion : float, optional
            Proportion of learn/verify data splitting 0.6 < x < 0.9. Default = 0.8.

        Returns
        -------
        None.

        """
        self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(dataHandler.xTrain(), dataHandler.yTrain(), train_size=proportion, random_state=0)
        
        self.learning_rate = lr
        self.estimators = estimators
        self.min_sample = min_sample
        
    def run(self):
        grad_boosting = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.estimators, min_samples_split=self.min_sample)
        grad_boosting.fit(self.X_learn, self.y_learn.ravel())
        print(grad_boosting.score(self.X_verify, self.y_verify))