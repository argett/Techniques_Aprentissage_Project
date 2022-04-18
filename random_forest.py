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
    def __init__(self, dataHandler, nb_trees=350, max_depth=25, min_sample=2, max_features=1, criterion="entropy", proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        nb_trees : int
            The number of trees in the forest, 10 < x < 1000
        criterion : {0="gini", 1="entropy"}, optional
            The number of trees in the forest. Default = 0 ("Gini")
        max_features : int, optional
            The maximum depth of the tree. Default = the number of features
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. Default = 0.2.

        Returns
        -------
        None.
        """
        self.randomForest = None
        self.dh = dataHandler
        
        self.proportion = proportion
        
        self.trees = nb_trees
        self.criterion = criterion         
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_sample = min_sample
        
        self.err_train = []
        self.err_valid = []
    
    def recherche_hyperparametres(self, num_fold):   
        """
        The function is going to try every possibility of combinaison within the given lists of parameters to find the one which has the less error on the model

        Parameters
        ----------
        num_fold : int
            The number of time the the k-cross validation is going to be made.
        nb_trees : list[float]
            List of all numbers of trees in the forest ot try.
        maxDepth : list[float] or None
            List of maximums depth of the tree to try. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        max_features : list[float]
           List of all number of features to try, to consider when looking for the best split.
        min_sample : list[float]
            List of all minimum numbers of samples to try, required to split an internal node.
        criterion : 0=“gini” or 1=“entropy”
            The function to measure the quality of a split.

        Returns
        -------
        None.
        """
        nb_trees = self.trees
        maxDepth = self.max_depth
        max_features = self.max_features
        min_sample = self.min_sample
        criterion = self.criterion
        
        liste_res = [] 
        liste_crit = [] 
        liste_tree = [] 
        liste_depth = []
        liste_feature = []
        liste_sample = []
        
        meilleur_res = 0
        meilleur_crit = None
        meilleur_tree = None
        meilleur_depth = None
        meilleur_feature = None
        meilleur_sample = None
        
        for crit in criterion:
            print(crit)
            for tree in tqdm(nb_trees):  # On teste plusieurs degrés du polynôme
                for depth in maxDepth:
                    for feature in max_features:
                        for sample in min_sample:
                            sum_result = 0
                            
                            self.trees = tree
                            self.max_features = feature
                            self.max_depth = depth
                            self.min_sample = sample
                            for k in range(num_fold):  # K-fold validation
                                X_learn, X_verify, y_learn, y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=k, shuffle=True)
                                self.entrainement(X_learn, y_learn)
                                self.err_valid.append(self.score(X_verify, y_verify))
                                #print("Avec k= " + str(k) + ", leaf_size = " + str(ls) + ", le score de verify est " + str(score))
                                sum_result += self.err_valid[-1]
                                
                            avg_res_locale = sum_result/(num_fold)  # On regarde la moyenne des erreurs sur le K-fold  
                            
                            liste_res.append(avg_res_locale)  
                            if crit == "gini":
                                liste_crit.append(0)
                            else:
                                liste_crit.append(1)
                            liste_tree.append(tree)  
                            liste_depth.append(depth)  
                            liste_feature.append(feature)  
                            liste_sample.append(sample)  
                     
                            if(avg_res_locale > meilleur_res):
                                meilleur_res = avg_res_locale
                                meilleur_crit = crit
                                meilleur_tree = tree
                                meilleur_depth = depth
                                meilleur_feature = feature
                                meilleur_sample = sample
                                    
        self.criterion = meilleur_crit
        self.trees = meilleur_tree
        self.max_features = meilleur_feature
        self.max_depth = meilleur_depth
        self.min_sample = meilleur_sample
        
        plt.plot(liste_res)  
        plt.title("Random Forest : Bonne réponse moyenne sur les K-fold validations")  
        plt.show()  
        plt.plot(liste_crit) 
        plt.title("Random Forest : Critère 0=gini, 1=entropie")  
        plt.show()   
        plt.plot(liste_tree)  
        plt.title("Random Forest : Nombre d'arbre dans la forêt")  
        plt.show()  
        plt.plot(liste_depth)  
        plt.title("Random Forest : Profondeur des arbres")  
        plt.show()   
        plt.plot(liste_feature)  
        plt.title("Random Forest : Nombre de caractéristiques nécéssaire pour séparer les données")  
        plt.show()  
        plt.plot(liste_sample)  
        plt.title("Random Forest : Nombre d'éléments minnimum pour pouvoir passer d'une feuille à un noeud")  
        plt.show()  
        
        print("criterion = " + str(meilleur_crit))
        print("trees = " + str(meilleur_tree))
        print("max_features = " + str(meilleur_feature))
        print("meilleur_depth = " + str(meilleur_depth))
        print("meilleur_sample = " + str(meilleur_sample))
        
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
        self.randomForest = RandomForestClassifier(n_estimators=self.trees, max_depth=self.max_depth, min_samples_split=self.min_sample,criterion=self.criterion, max_features=self.max_features, n_jobs=-1)
        self.randomForest.fit(xData, yData.ravel())
        self.err_train.append(self.score(xData, yData.ravel()))
    
    def score(self, xData, yData):
        """
        Take the fitted model to check on the validation dataset.

        Parameters
        ----------

        Returns
        -------
        float
            The score of the model.
        """
        return self.randomForest.score(xData, yData) 
    
    def run(self):
        """
        Run the model on pre-loaded testing data.

        Returns
        -------
        List[string]
            Every class or label deduced from the entry dataset with the trained model
        """
        return self.randomForest.predict(self.dh.xUnknownData())
