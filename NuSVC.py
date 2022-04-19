# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:55:10 2022

@author: Tsiory
"""
from GeneralModel import CommonModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.svm import NuSVC


class NUSVC(CommonModel):
    def __init__(self, dataHandler, kernel='rbf', gamma='scale', proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        kernel : str
            The kernel to use
        gamma : float
            The value of gamma
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The
            default is 0.2.

        Returns
        -------
        None.

        """
        CommonModel.__init__(self,dataHandler, proportion)
        self.NSVC = None

        self.kernel = kernel
        self.gamma = gamma

    def recherche_hyperparametres(self, num_fold):
        """
        Recherche des hyperparamètres pour le NuSVC

        Parameters
        ----------
        num_fold : int
            The number of folds
        kernel : str
            The kernel to use
        gamma : float
            The value of gamma

        Returns
        -------
        None.

        """
        liste_res = []
        liste_kernel = []
        liste_gamma = []

        meilleur_kernel = None
        meilleur_gamma = None

        for k in tqdm(self.getListParameters(0)):
            for g in self.getListParameters(1):
                sum_results = 0

                self.kernel = k
                self.gamma = g

                for i in range(num_fold):
                    X_train, X_test, y_train, y_test = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=i, shuffle=True)
                    self.entrainement(X_train, y_train)
                    self.err_valid.append(self.score(X_test, y_test))
                    sum_results += self.err_valid[-1]

                avg_res_local = sum_results / num_fold

                liste_res.append(avg_res_local)
                liste_kernel.append(k)
                liste_gamma.append(g)

                if avg_res_local > self.betterValidationScore:
                    self.betterValidationScore = avg_res_local
                    meilleur_kernel = k
                    meilleur_gamma = g

        self.kernel = meilleur_kernel
        self.gamma = meilleur_gamma

        plt.plot(liste_res)
        plt.title("NuSVC : Bonne réponse moyenne sur les " + str(num_fold) + " folds")
        plt.show()

        plt.plot(liste_kernel)
        plt.title("NuSVC : Kernel")
        plt.show()

        plt.plot(liste_gamma)
        plt.title("NuSVC : Gamma")
        plt.show()

        print("\nLes meilleurs parametres parmis ceux essayes sont :")
        print("\tMeilleur kernel : " + str(meilleur_kernel))
        print("\tMeilleur gamma : " + str(meilleur_gamma))

    def entrainement(self, X_train, y_train):
        """
        Entrainement du NuSVC

        Parameters
        ----------
        X_train : dataframe
            Training data
        y_train : list
            Training target

        Returns
        -------
        None.

        """
        if self.kernel == 'rbf' or self.kernel == 'poly' or self.kernel == 'sigmoid':
            self.NSVC = NuSVC(nu=0.2, kernel=self.kernel, gamma=self.gamma)
        else:
            self.NSVC = NuSVC(nu=0.2, kernel=self.kernel)
        self.NSVC.fit(X_train, y_train.ravel())
        self.err_train.append(self.score(X_train, y_train.ravel()))

    def score(self, X_test, y_test):
        """
        Calcul du score du NuSVC

        Parameters
        ----------
        X_test : dataframe
            Test data
        y_test : list
            Test target

        Returns
        -------
        float
            The score

        """
        return self.NSVC.score(X_test, y_test)

    def run(self):
        """
        Run the NuSVC

        Parameters
        ----------
        None.

        Returns
        -------
        List[str]
            The list of the predicted classes

        """
        return self.NSVC.predict(self.dh.xUnknownData())
