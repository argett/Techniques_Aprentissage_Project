# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:21:33 2022

@author: Lilian
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from tqdm import tqdm


class gradientBoosting():
    def __init__(self, dataHandler, lr=0.0001, estimators=100, min_sample=2, proportion=0.2):
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
        
        self.gradientBoosting = None
        self.dh = dataHandler
        
        self.proportion = proportion        
        self.learning_rate = lr
        self.estimators = estimators
        self.min_sample = min_sample
        
    def recherche_hyperparametres(self, num_fold, learning_rate, n_estimators, min_samples_split): 
        meilleur_err = np.inf 
        meilleur_lr = None 
        meilleur_estimantor = None 
        meilleur_sample = None 
         
        for lr in tqdm(learning_rate):  # On teste plusieurs degrés du polynôme 
            for esti in n_estimators: 
                for samp in min_samples_split: 
                            sum_error = 0 
                             
                            self.learning_rate = lr
                            self.estimators = esti
                            self.min_sample = samp
                 
                            for k in range(num_fold):  # K-fold validation 
                                self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=k, shuffle=True) 
                                sum_error += self.entrainement()                                     
                                 
                            avg_err_locale = sum_error/(num_fold)  # On regarde la moyenne des erreurs sur le K-fold   
                            if(avg_err_locale < meilleur_err): 
                                meilleur_err = avg_err_locale 
                                meilleur_lr = lr 
                                meilleur_estimantor = esti 
                                meilleur_sample = samp 
                                     
        self.learning_rate = meilleur_lr
        self.estimators = meilleur_estimantor
        self.min_sample = meilleur_sample
          
        print("meilleur_lr = " + str(meilleur_lr)) 
        print("meilleur_estimantor = " + str(meilleur_estimantor)) 
        print("meilleur_sample = " + str(meilleur_sample)) 
     
    def entrainement(self): 
        self.gradientBoosting = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.estimators, min_samples_split=self.min_sample)
        self.gradientBoosting.fit(self.X_learn, self.y_learn.ravel())
        return self.gradientBoosting.score(self.X_verify, self.y_verify) 
     
    def run(self): 
        print(self.gradientBoosting.predict(self.dh.xTest())) 