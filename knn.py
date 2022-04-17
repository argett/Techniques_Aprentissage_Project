# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:53:11 2022

@author: Lilian
""" 

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
 
 
class knn():
    def __init__(self, dataHandler, k=3, leaf_size=2, proportion=0.2, manhattan=True):
        """ 
        Create an instance of the class 
 
        Parameters 
        ---------- 
        dataHandler : Dataset 
            The dataset with everything loaded from the main
        k : int, optional
            The number of neighbour we want to take into account in the KNN algorithm. The default is 8. 
        leaf_size : int, optional 
            Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. The default is 64. 
        proportion : float, optional 
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The default is 0.2. 
        manhattan : boolean, optional 
            Select the Euclidean distance by default, select the Manhattan distance if True. The default is False. 
 
        Returns 
        ------- 
        None. 
 
        """ 
        self.nn = None
        self.dh = dataHandler
        
        self.proportion = proportion
        
        self.nb_neighbour = k 
        self.leaf_size = leaf_size
        self.distance_method = manhattan # euclidean distance by default     
        
        self.err_train = []
        self.err_valid = []

    def recherche_hyperparametres(self, num_fold, number_cluster, leaf):
        """
        The function is going to try every possibility of combinaison within the given lists of parameters to find the one which has the less error on the model.

        Parameters
        ----------
        num_fold : int
            The number of time the the k-cross validation is going to be made.
        number_cluster : list[int]
            List of all number of neighbors to use by default for kneighbors queries to try.
        leaf : list[int]
            List of all leafs size passed to BallTree or KDTree to try. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

        Returns
        -------
        None.

        """
        liste_res = [] 
        liste_k = [] 
        liste_leaf = [] 
        
        meilleur_res = 0
        meilleur_k = None 
        meilleur_leaf = None 
         
        for k in tqdm(number_cluster):  # On teste plusieurs degrés du polynôme 
            for ls in leaf:
                sum_result = 0 
                 
                self.nb_neighbour = k
                self.leaf_size = ls
     
                for ki in range(num_fold):  # K-fold validation 
                    X_learn, X_verify, y_learn, y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=ki, shuffle=True) 
                    self.entrainement(X_learn, y_learn)
                    self.err_valid.append(self.score(X_verify, y_verify))
                    #print("Avec k= " + str(k) + ", leaf_size = " + str(ls) + ", le score de verify est " + str(score))
                    sum_result += self.err_valid[-1]
                     
                avg_res_locale = sum_result/(num_fold)  # On regarde la moyenne des erreurs sur le K-fold   
                
                liste_res.append(avg_res_locale) 
                liste_k.append(k) 
                liste_leaf.append(ls) 
                    
                if(avg_res_locale > meilleur_res): 
                    meilleur_res = avg_res_locale 
                    meilleur_k = k
                    meilleur_leaf = ls
                
        self.nb_neighbour = meilleur_k
        self.leaf_size = meilleur_leaf
        
        plt.plot(liste_res) 
        plt.title("KNN : Bonne réponse moyenne sur les K-fold validations") 
        plt.show() 
        plt.plot(liste_k) 
        plt.title("KNN : Nombre de voisins ou K") 
        plt.show() 
        plt.plot(liste_leaf) 
        plt.title("KNN : Valeurs de la taille des feuilles") 
        plt.show() 
        print("meilleur_k = " + str(meilleur_k) + " et meilleur leaf_size = " + str(meilleur_leaf))

    def entrainement(self, xData, yData):
        """
        Fit the model with respect to the parameters given by the k-fold function or the ones given when initialising the model.

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
        self.nn = KNeighborsClassifier(n_neighbors=self.nb_neighbour, leaf_size=self.leaf_size, n_jobs=-1) 
        self.nn.fit(xData, yData.ravel())
        self.err_train.append(self.score(xData, yData.ravel()))
    
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
            Every class or label deduced from the entry dataset with the trained model
        """
        return self.nn.predict(self.dh.xUnknownData())
