# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:56:05 2022

@author: Tsiory
"""
from GeneralModel import CommonModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.svm import SVC


class svc(CommonModel):
    def __init__(self, dataHandler, C=1.0, kernel='rbf', gamma='scale', proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        C : float, optional
            Regularization parameter [1, 100]. Defaults to 1.0.
        kernel : str
            Specifies the kernel type to be used in the algorithm. Defaults to
            'rbf'.
        gamma : str
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
            Defaults to 'scale'.
        proportion : float
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The
            default is 0.2.

        Returns
        -------
        None.

        """
        CommonModel.__init__(self, dataHandler, proportion)
        self.Svc = None

        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def recherche_hyperparametres(self, num_fold):
        """
        Recherche des hyperparamètres pour le SVC

        Parameters
        ----------
        num_fold : int
            Number of folds for cross validation
        C : float
            Regularization parameter
        kernel : str
            Specifies the kernel type to be used in the algorithm
        gamma : str
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels

        Returns
        -------
        None.

        """
        liste_resultats = []
        liste_C = []
        liste_gamma = []
        liste_kernel = []

        better = False
        meilleur_C = None
        meilleur_gamma = None
        meilleur_kernel = None

        for c in tqdm(self.getListParameters(0)):
            for g in self.getListParameters(2):
                for k in self.getListParameters(1):
                    sum_results = 0

                    self.C = c
                    self.gamma = g
                    self.kernel = k

                    for i in range(num_fold):
                        X_train, X_test, y_train, y_test = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=i, shuffle=True)
                        self.entrainement(X_train, y_train)
                        self.err_valid.append(self.score(X_test, y_test))
                        sum_results += self.err_valid[-1]

                    avg_res_local = sum_results / num_fold

                    liste_resultats.append(avg_res_local)
                    liste_C.append(c)
                    liste_gamma.append(g)
                    liste_kernel.append(k)

                    if avg_res_local > self.betterValidationScore:
                        better = True
                        self.betterValidationScore = avg_res_local
                        meilleur_C = c
                        meilleur_gamma = g
                        meilleur_kernel = k

        if better:
            self.C = meilleur_C
            self.gamma = meilleur_gamma
            self.kernel = meilleur_kernel

            print("\nLes meilleurs parametres parmis ceux essayes sont :")
            print("\tMeilleur C : ", meilleur_C)
            print("\tMeilleur Gamma : ", meilleur_gamma)
            print("\tMeilleur Kernel : ", meilleur_kernel)

        plt.plot(liste_resultats)
        plt.title("SVC : Bonne réponse moyenne en fonction du nombre de folds")
        plt.show()

        plt.plot(liste_C)
        plt.title("SVC : C en fonction du nombre de folds")
        plt.show()

        plt.plot(liste_gamma)
        plt.title("SVC : Gamma en fonction du nombre de folds")
        plt.show()

        plt.plot(liste_kernel)
        plt.title("SVC : Kernel en fonction du nombre de folds")
        plt.show()

    def entrainement(self, X_train, y_train):
        """
        Entrainement du SVC

        Parameters
        ----------
        X_train : dataframe
            Training data
        y_train : list
            Target values

        Returns
        -------
        None.

        """
        if self.kernel == 'rbf' or self.kernel == 'poly' or self.kernel == 'sigmoid':
            self.Svc = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        else:
            self.Svc = SVC(C=self.C, kernel=self.kernel)
        self.Svc.fit(X_train, y_train.ravel())
        self.err_train.append(self.score(X_train, y_train))

    def score(self, X_test, y_test):
        """
        Calcul du score du SVC

        Parameters
        ----------
        X_test : dataframe
            Test data
        y_test : list
            Target values

        Returns
        -------
        float
            Score du SVC

        """
        return self.Svc.score(X_test, y_test)

    def run(self):
        """
        Run the SVC

        Parameters
        ----------
        None.

        Returns
        -------
        List[str]
            The list of the predicted classes

        """
        return self.Svc.predict(self.dh.xUnknownData())
