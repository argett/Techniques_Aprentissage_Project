# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:54:11 2022

@author: Tsiory
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from tqdm import tqdm
# import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import ShuffleSplit


class LDA():
    def __init__(self, dataHandler, solver='svd', shrinkage=None, proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        solver : str
            Solver
        shrinkage : str or float
            Shrinkage if solver is 'lsqr' or 'eigen'
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The
            default is 0.2.

        Returns
        -------
        None.

        """
        self.lda = None
        self.dh = dataHandler

        self.proportion = proportion

        self.solver = solver
        self.shrinkage = shrinkage

        self.err_train = []
        self.err_valid = []

    def recherche_hyperparametres(self, num_fold, solver, shrinkage):
        """
        Hyperparameter search

        Parameters
        ----------
        num_fold : int
            Number of folds
        solver : str
            Solver
        shrinkage : str or float
            Shrinkage if solver is 'lsqr' or 'eigen'

        Returns
        -------
        None.

        """
        liste_res = []
        liste_solver = []
        liste_shrinkage = []

        meilleur_res = 0
        meilleur_solver = None
        meilleur_shrinkage = None

        for solv in tqdm(solver):
            for sh in shrinkage:
                sum_results = 0

                self.solver = solv
                self.shrinkage = sh

                for i in range(num_fold):
                    X_train, X_test, y_train, y_test = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=i, shuffle=True)
                    self.entrainement(X_train, y_train)
                    self.err_valid.append(self.score(X_test, y_test))
                    sum_results += self.err_valid[-1]

                avg_res_locale = sum_results / num_fold

                liste_res.append(avg_res_locale)
                liste_solver.append(solv)
                liste_shrinkage.append(sh)

                if avg_res_locale > meilleur_res:
                    meilleur_res = avg_res_locale
                    meilleur_solver = solv
                    meilleur_shrinkage = sh

        self.solver = meilleur_solver
        self.shrinkage = meilleur_shrinkage

        plt.plot(liste_res)
        plt.title("Linear Discriminant Analysis : Bonne réponse moyenne sur les K validations")
        plt.show()

        plt.plot(liste_solver)
        plt.title("Linear Discriminant Analysis : Solver")
        plt.show()

        plt.plot(liste_shrinkage)
        plt.title("Linear Discriminant Analysis : Shrinkage")
        plt.show()

        print("Meilleur solver : ", meilleur_solver)
        print("Meilleur shrinkage : ", meilleur_shrinkage)

    def entrainement(self, X_train, y_train):
        """
        Train the model

        Parameters
        ----------
        X_train : dataframe
            Training data
        y_train : list
            Training labels

        Returns
        -------
        None.

        """
        if self.solver == 'svd':
            self.lda = LinearDiscriminantAnalysis(solver=self.solver)
        else:
            self.lda = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
        self.lda.fit(X_train, y_train.ravel())
        self.err_train.append(self.score(X_train, y_train.ravel()))

    def score(self, X_test, y_test):
        """
        Compute the score

        Parameters
        ----------
        X_test : dataframe
            Test data
        y_test : list
            Test labels

        Returns
        -------
        float
            Score

        """
        return self.lda.score(X_test, y_test)

    def run(self):
        """
        Run the model

        Parameters
        ----------
        None.

        Returns
        -------
        List[string]
            Every class or label deduced from the entry dataset with the
            trained model

        """
        return self.lda.predict(self.dh.xUnknownData())
