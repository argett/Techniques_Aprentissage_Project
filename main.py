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
    arg3 = 0.85  # What is the max % of caracteristics similar in a 10% range with respect to the total range of the caracteristic
    arg4 = 0     # Boolean to allow cross-validation or not
    arg5 = 2     # Number of k in the k-cross validation
    
    # KNN
    arg6 = 3  # Number of K in KNN algorithm
    arg7 = 2 # Number of leaf size
    
    # random forest parameters
    arg8 = 350 # Number of threes in the random forest
    arg9 = 50 # maximum depth of the tress
    arg10 = 2 # Number of minimum samples to create a new node
    arg11 = 50 # Controls both the randomness of the bootstrapping of the samples used when building trees
    arg12 = 25 #The number of features to consider when looking for the best split

    # Gradient Boosting
    arg13 = 0.1  # learning rate
    arg14 = 400  # number of estimator
    arg15 = 2  # Minimum number of sample to create a new node
    
    """
    if len(sys.argv) < 8:
        print("Usage: python main.py sk dataPath\n")

        print("\t path: The path to the Data folder")
        print("\t display caracteristics: The path to the Data folder")
        print("\t feature selection: The maximum % of carecteristics in a 10% range of total caracteristics values [0.3,1]")
        print("\t Number of neighbour for KNN: The number of neighbour in the KNN-calssifyer")
        print(" exemple: python3 main.py ../Data\n")
        return

    path_data = str(sys.argv[1])
    """
    dataset = dt.Dataset(arg1, arg2, arg3) 
    model_knn = knn.knn(dataset, arg6, arg7)
    model_rdmForest = rForest.randomForest(dataset, arg8, arg9, arg10, arg11, arg12)
    model_gradBoost = gradientB.gradientBoosting(dataset, arg13, arg14, arg15) 
    
    if arg4 == 1:  # Cross validation   
        kFold = arg5
        
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

        # Gradient Boosting
        lr = [0.01,0.05,0.1,0.5]
        estimator = [10,100,400,500]
        sample = [2]
        model_gradBoost.recherche_hyperparametres(kFold, lr, estimator, sample) 

    print(model_knn.entrainement())
    res1 = model_knn.run()
    
    print(model_rdmForest.entrainement())
    res2 = model_rdmForest.run()
    """
    print(model_gradBoost.entrainement())
    res3 = model_gradBoost.run()
    """
    processResult([res1.tolist(), res2.tolist()])