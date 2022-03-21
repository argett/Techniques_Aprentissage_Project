# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:48:20 2022

@author: Lilian
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class randomForest():
    def __init__(self, dataHandler, nb_trees=350, max_depth=50, min_sample=2, random_state=50, max_features=25, criterion="gini", proportion=0.2):
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
        
        self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(dataHandler.xTrain(), dataHandler.yTrain(), test_size=proportion, random_state=0) 
        
        self.dh = dataHandler
        self.proportion = proportion
        
        self.trees = nb_trees
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.rdm_state = random_state
        self.min_sample = min_sample
    
    def recherche_hyperparametres(self, num_fold, nb_trees, maxDepth, random_state, max_features, min_sample, criterion):        
        """
        The function is going to try every possibility of combinaison within the given lists of parameters to find the one which has the less error on the model

        Parameters
        ----------
        num_fold : int
            The number of time the the k-cross validation is going to be made.
        nb_trees : list[float]
            List of all numbers of trees in the forest ot try.
        maxDepth : list[float] or None
            List of maximums depth of the tree to try. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples..
        random_state : list[float]
            List of values to try that controls both the randomness of the bootstrapping of the samples used when building trees
        max_features : list[float]
           List of all number of features to try, to consider when looking for the best split.
        min_sample : list[float]
            List of all minimum numbers of samples to try, required to split an internal node.
        criterion : “gini” or “entropy”
            The function to measure the quality of a split.

        Returns
        -------
        None.

        """
        liste_res = [] 
        liste_tree = [] 
        liste_depth = []
        liste_state = []
        liste_feature = []
        liste_sample = []
        
        meilleur_res = 0
        meilleur_tree = None
        meilleur_depth = None
        meilleur_state = None
        meilleur_feature = None
        meilleur_sample = None
        
        for tree in tqdm(nb_trees):  # On teste plusieurs degrés du polynôme
            for depth in tqdm(maxDepth):
                for state in random_state:
                    for feature in max_features:
                        for sample in min_sample:
                            sum_result = 0
                            
                            self.trees = tree
                            self.max_features = feature
                            self.max_depth = depth
                            self.rdm_state = state
                            self.min_sample = sample
                            self.criterion = criterion
                
                            for k in range(num_fold):  # K-fold validation
                                self.X_learn, self.X_verify, self.y_learn, self.y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=k, shuffle=True)
                                sum_result += self.entrainement()                                
                                
                            avg_res_locale = sum_result/(num_fold)  # On regarde la moyenne des erreurs sur le K-fold  
                            
                            liste_res.append(avg_res_locale)  
                            liste_tree.append(tree)  
                            liste_depth.append(depth)  
                            liste_state.append(state)  
                            liste_feature.append(feature)  
                            liste_sample.append(sample)  
                     
                            if(avg_res_locale > meilleur_res):
                                meilleur_res = avg_res_locale
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
        
        plt.plot(liste_res)  
        plt.title("Random Forest : Bonne réponse moyenne sur les K-fold validations")  
        plt.show()  
        plt.plot(liste_tree)  
        plt.title("Random Forest : Nombre d'arbre dans la forêt")  
        plt.show()  
        plt.plot(liste_depth)  
        plt.title("Random Forest : Profondeur des arbres")  
        plt.show()  
        plt.plot(liste_state)  
        plt.title("Random Forest : Nombre qui contrôle le boostrapping pour la construction d'arbre")  
        plt.show()  
        plt.plot(liste_feature)  
        plt.title("Random Forest : Nombre de caractéristiques nécéssaire pour séparer les données")  
        plt.show()  
        plt.plot(liste_sample)  
        plt.title("Random Forest : Nombre d'éléments minnimum pour pouvoir passer d'une feuille à un noeud")  
        plt.show()  
        
        print("trees = " + str(meilleur_tree))
        print("max_features = " + str(meilleur_feature))
        print("meilleur_depth = " + str(meilleur_depth))
        print("meilleur_state = " + str(meilleur_state))
        print("meilleur_sample = " + str(meilleur_sample))
        
    def entrainement(self):
        """
        Fit the model with respect to the parameters given by the k-fold function or the ones given when initialising the model.

        Parameters
        ----------

        Returns
        -------
        float
            The score of the model.
        """
        self.randomForest = RandomForestClassifier(n_estimators=self.trees, max_depth=self.max_depth, min_samples_split=self.min_sample,criterion=self.criterion, max_features=self.max_features, n_jobs=-1)
        self.randomForest.fit(self.X_learn, self.y_learn.ravel()) # on utilise toutes les données d'entrainement
        return self.validate()
    
    def validate(self):
        """
        Take the fitted model to check on the validation dataset.

        Parameters
        ----------

        Returns
        -------
        float
            The score of the model.
        """
        return self.randomForest.score(self.X_verify, self.y_verify) 
    
    def run(self):
        """
        Run the model on pre-loaded testing data.

        Returns
        -------
        List[string]
            Every class or label deduced from the entry dataset with the trained model
        """
        return self.randomForest.predict(self.dh.xTest())
