# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:53:11 2022

@author: Lilian
"""
from GeneralModel import CommonModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Knn(CommonModel):
    def __init__(self, dataHandler, k=3, leafSize=2, proportion=0.2, manhattan=True):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        k : int, optional
            The number of neighbour we want to take into account in the KNN
            algorithm. The default is 8.
        leafSize : int, optional
            Leaf size passed to BallTree or KDTree. This can affect the speed
            of the construction and query, as well as the memory required to
            store the tree. The optimal value depends on the nature of the
            problem. The default is 64.
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The
            default is 0.2.
        manhattan : boolean, optional
            Select the Euclidean distance by default, select the Manhattan
            distance if True. The default is False.

        Returns
        -------
        None.

        """
        CommonModel.__init__(self, dataHandler, proportion)
        self.nn = None

        self.nbNeighbour = k
        self.leafSize = leafSize
        self.distanceMethod = manhattan  # euclidean distance by default

    def recherche_hyperparametres(self, numFold):
        """
        The function is going to try every possibility of combinaison within
        the given lists of parameters to find the one which has the less error
        on the model.

        Parameters
        ----------
        numFold : int
            The number of time the the k-cross validation is going to be made.
        number_cluster : list[int]
            List of all number of neighbors to use by default for kneighbors
            queries to try.
        leaf : list[int]
            List of all leafs size passed to BallTree or KDTree to try. This
            can affect the speed of the construction and query, as well as the
            memory required to store the tree. The optimal value depends on
            the nature of the problem.

        Returns
        -------
        None.

        """
        listeRes = []
        listeDist = []
        listeK = []
        listeLeaf = []

        better = False
        meilleurDist = None
        meilleurK = None
        meilleurLeaf = None

        for dis in self.getListParameters(2):
            # On teste plusieurs degrés du polynôme
            for k in tqdm(self.getListParameters(0)):
                for ls in self.getListParameters(1):
                    sumResult = 0

                    self.distanceMethod = dis
                    self.nbNeighbour = k
                    self.leafSize = ls

                    for ki in range(numFold):  # K-fold validation 
                        X_learn, X_verify, y_learn, y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=ki, shuffle=True)
                        self.entrainement(X_learn, y_learn)
                        self.errValid.append(self.score(X_verify, y_verify))
                        sumResult += self.errValid[-1]

                    # On regarde la moyenne des erreurs sur le K-fold
                    avg_res_locale = sumResult/(numFold)

                    listeRes.append(avg_res_locale)
                    listeDist.append(dis)
                    listeK.append(k)
                    listeLeaf.append(ls)

                    if(avg_res_locale > self.betterValidationScore):
                        better = True
                        self.betterValidationScore = avg_res_locale
                        meilleurDist = dis
                        meilleurK = k
                        meilleurLeaf = ls

        if better:
            self.distanceMethod = meilleurDist
            self.nbNeighbour = meilleurK
            self.leafSize = meilleurLeaf

            print("\nLes meilleurs parametres parmis ceux essayes sont :")
            print("\tMeilleur nombre de voisins = " + str(meilleurK))
            print("\tMeilleur taille de feuille = " + str(meilleurLeaf))
            print("\tMeilleure distance (0=euclidienne, 1=manhattan)= " + str(meilleurDist))

        plt.plot(listeRes)
        plt.title("KNN : Bonne réponse moyenne sur les K-fold validations")
        plt.show()
        plt.plot(listeDist)
        plt.title("KNN : Distance (0=euclidienne, 1=manhattan)")
        plt.show()
        plt.plot(listeK)
        plt.title("KNN : Nombre de voisins ou K")
        plt.show()
        plt.plot(listeLeaf)
        plt.title("KNN : Valeurs de la taille des feuilles")
        plt.show()

    def entrainement(self, xData, yData):
        """
        Fit the model with respect to the parameters given by the k-fold
        function or the ones given when initialising the model.

        Parameters
        ----------
        xData : 2D array, dataframe
            Array of the dataset.
        yData : 1D array, list
            Class corresponding to the dataset.

        Returns
        -------
        None.

        """
        self.nn = KNeighborsClassifier(n_neighbors=self.nbNeighbour, leaf_size=self.leafSize, n_jobs=-1)
        self.nn.fit(xData, yData.ravel())
        self.errTrain.append(self.score(xData, yData.ravel()))

    def score(self, xData, yData):
        """
        Compute the score of the model.

        Parameters
        ----------

        Returns
        -------
        float
            The score of the model.
        """
        return self.nn.score(xData, yData)

    def run(self):
        """
        Run the model on pre-loaded testing data.

        Returns
        -------
        List[string]
            Every class or label deduced from the entry dataset with the
            trained model
        """
        return self.nn.predict(self.dh.xUnknownData())
