# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:54:11 2022

@author: Tsiory
"""
from GeneralModel import CommonModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(CommonModel):
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
        CommonModel.__init__(self,dataHandler, proportion)
        self.lda = None

        self.solver = solver
        self.shrinkage = shrinkage

    def recherche_hyperparametres(self, num_fold):
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
        
        better = False
        meilleur_solver = None
        meilleur_shrinkage = None

        for solv in tqdm(self.getListParameters(0)):
            for sh in self.getListParameters(1):
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

                if avg_res_locale > self.betterValidationScore:
                    better = True
                    self.betterValidationScore = avg_res_locale
                    meilleur_solver = solv
                    meilleur_shrinkage = sh

        if better :
            self.solver = meilleur_solver
            self.shrinkage = meilleur_shrinkage

            print("\nLes meilleurs parametres parmis ceux essayes sont :")
            print("\tMeilleur solver : ", meilleur_solver)
            print("\tMeilleur shrinkage : ", meilleur_shrinkage)

        plt.plot(liste_res)
        plt.title("Linear Discriminant Analysis : Bonne réponse moyenne sur les K validations")
        plt.show()

        plt.plot(liste_solver)
        plt.title("Linear Discriminant Analysis : Solver")
        plt.show()

        plt.plot(liste_shrinkage)
        plt.title("Linear Discriminant Analysis : Shrinkage")
        plt.show()

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
