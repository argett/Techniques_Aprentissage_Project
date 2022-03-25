# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:40:02 2022

@author: Lilian FAVRE GARCIA
         Andrianihary Tsiory RAZAFINDRAMISA

Le projet a pour objectif de tester au moins six méthodes de classification
sur une base de données Kaggle (www.kaggle.com) avec la bibliothèque
scikit-learn (https://scikit-learn.org). Les équipes sont libres de choisir
la base de données de leur choix, mais une option simple est celle du
challenge de classification de feuilles d’arbres. Pour ce projet, on s’attend
à ce que les bonnes pratiques de cross-validation et de recherche
d’hyper-paramètres soient mises de l’avant pour identifier la meilleure
solution possible pour résoudre le problème.
https://www.kaggle.com/c/leaf-classification

respecter le standard pep8 (https://www.python.org/dev/peps/pep-0008/)
et respecter un standard uniforme pour la nomenclature des variables,
des noms de fonctions et des noms de classes. Évitez également les variables
« hardcodées » empêchant l’utilisation de votre programme sur un autre
ordinateur que le vôtre.

Choix de design : vous devez organiser votre code de façon professionnelle.
Pour ce faire, on s’attend à une hiérarchie de classes cohérente, pas
seulement une panoplie de fonctions disparates. Aussi, du code dans un script
« qui fait tout » se verra automatiquement attribuer la note de zéro.
Bien que non requis, on vous encourage à faire un design de classes avant
de commencer à coder et à présenter un diagramme de classe dans votre rapport.
Aussi, le code, les données et la documentation doivent être organisés suivant
une bonne structure de répertoires.

Démarche scientifique :
    Pour ce volet, vous devez vous poser les questions suivantes :
avez-vous bien « cross-validé » vos méthodes? Avez-vous bien fait votre
recherche d’hyper-paramètres? Avez-vous entraîné et testé vos méthodes sur
les mêmes données? Est-ce que cela transparaît dans le rapport? Avez-vous
uniquement utilisé les données brutes ou avez-vous essayé de les réorganiser
pour améliorer vos résultats? Etc.
"""
# gridsearch CV verbose = 1.2.3... n_jobs=-1
import matplotlib.pyplot as plt 
import Dataset as dt
import random_forest as rForest
import knn as knn
import gradient_boosting as gradientB
import numpy as np
  
def processResult(allLists):
    nbList = len(allLists)
    if(nbList <= 1):
        print("No enought models to compare the results")
        return
    
    list_compare = np.zeros(nbList, dtype=int)
    for i in range(len(allLists[0])):        
        classes = []
        for j in range(nbList):
            classes.append(allLists[j][i])
            
        mostCommon_label = max(classes,key=classes.count)
        
        for j in range(nbList):
            if allLists[j][i] == mostCommon_label:
                list_compare[j] += 1   
    
    """
    plot illissible
    plt.figure(figsize=(100, 30))
    for liste in allLists:
        plt.plot(liste)
    plt.show()
    """
    for i in range(nbList):
        print("La liste " + str(i) + " a " + str(list_compare[i]) + " / " + str(len(allLists[0])) + " résultats qui sont comme la réponse la plus fréquente")
    
    
    
if __name__ == "__main__":
    arg1 = "Data/"  # Path of the data folder
    # TODO : arg2 = "knn" "gradBoost" "rdmForest"
    arg2 = False  # Display caracteristics histograms
    arg3 = 1  # What is the max % of caracteristics similar in a 10% range with respect to the total range of the caracteristic    
    arg4 = -1 # Number of K for Train/Test split kcross-validation. -1 for no kcross validation
    
    arg5 = 0     # Boolean to allow the hyperparameters research or not
    arg6 = 1     # Number of k in the k-cross validation
    
    # KNN
    arg7 = 3  # Number of K in KNN algorithm
    arg8 = 2  # Number of leaf size
    
    # random forest parameters
    arg9 = 350 # Number of threes in the random forest
    arg10 = 50 # maximum depth of the trees
    arg11 = 2 # Number of minimum samples to create a new node
    arg12 = 50 # Controls both the randomness of the bootstrapping of the samples used when building trees
    arg13 = 25 #The number of features to consider when looking for the best split

    # Gradient Boosting
    arg14 = 0.1  # learning rate
    arg15 = 400  # number of estimator
    arg16 = 2  # Minimum number of sample to create a new node
    
    """
    if len(sys.argv) < 8:
        print("Usage: python main.py ... and add the parameters\n")
        print("There are 4 mandatory parameters : \n")
        
        print("\t Path: The path to the Data folder, string")
        print("\t Display caracteristic (0 or 1), to display or not the selected histograms")
        print("\t Feature selection: The maximum % of carecteristics in a 10% range of total caracteristics values [0.3,1], float
             \n\t\t 1 = select all data, 0.85 = don't take every caracteristic where 85% of the data is in the 10% range of the total range")
        print("\t Number of K for Train/Test split kcross-validation. min 3 or -1 for no kcross validation, int)
        print("\t Allow the hyperparameter research (0 or 1): 0 = take the default values, 1 = apply the hyperparameter research")
        print("\t -only if you allowed the hyperparameter research- Number of K in the kcross validation: [1,10] (10 can be long to compute for gradient boosting for example), integer")
             
        print("The following is going to depend on the type of model you want :")
        
        print("\n\t\t --- KNN model ---")
        print("\t Number of neighbour for KNN: The number of neighbour in the KNN-calssifyer. Minimum 1, integer")
        print("\t Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. Minimum 1, integer")
        
        
        print("\n\t\t --- Random Forest model ---")
        print("\t Number of threes in the random forest. [10,1000], integer")
        print("\t Maximum depth of the tress. Minimum 1, integer")
        print("\t Number of minimum samples to create a new node. Minimum 1, integer")
        print("\t Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). Minimum 1, integer")
        print("\t The number of features to consider when looking for the best split. Minimum 1, integer")
        
        
        print("\n\t\t --- Gradient Boosting model ---")
        print("\t Learning rate. [0.0001,1], float")
        print("\t The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance. [10,1000], integer")
        print("\t The minimum number of samples required to split an internal node. [1,98], integer")
        
        
        print(" exemple: python3 main.py ../Data\n")
        return

    path_data = str(sys.argv[1])
    """
    
    dataset = dt.Dataset(arg1, arg2, arg3, arg4) 
    model_knn = knn.knn(dataset, arg7, arg8)
    model_rdmForest = rForest.randomForest(dataset, arg9, arg10, arg11, arg12, arg13)
    model_gradBoost = gradientB.gradientBoosting(dataset, arg14, arg15, arg16) 
    
    if arg5 == 1:  # Cross validation   
        kFold = arg6
        
        # KNN
        ks = [3,5,8,10,15,20,30,50,80,100] 
        ls = [2,5,10,25,50,64,99] 
        model_knn.recherche_hyperparametres(kFold, ks, ls)  
        
        # Random Forest
        nb_trees = [50,200,350,500,800]
        maxDepth = [25,32,50,64,100]
        random_state = [10,25,50,75,100]
        max_features = [1,5,10,15,20,25,30]
        min_sample = [2]
        criterion = "gini"
        model_rdmForest.recherche_hyperparametres(kFold, nb_trees, maxDepth, random_state, max_features, min_sample, criterion)
        """
        # Gradient Boosting
        lr = [0.01,0.05,0.1,0.5]
        estimator = [10,100,400,500]
        sample = [2]
        model_gradBoost.recherche_hyperparametres(kFold, lr, estimator, sample) 
        """
    print("KNN :")
    model_knn.entrainement(dataset.xTrain(), dataset.yTrain())
    print(model_knn.score(dataset.xTest(), dataset.yTest()))
    res1 = model_knn.run()
    
    print("Random forest :")
    model_rdmForest.entrainement(dataset.xTrain(), dataset.yTrain())
    print(model_rdmForest.score(dataset.xTest(), dataset.yTest()))
    res2 = model_rdmForest.run()
    """
    print("Gradient boosting :")
    model_gradBoost.entrainement(dataset.xTrain(), dataset.yTrain())
    print(model_gradBoost.score(dataset.xTest(), dataset.yTest()))
    res3 = model_gradBoost.run()
    """
    processResult([res1.tolist(), res2.tolist(), res3.tolist()])