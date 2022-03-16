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

import Dataset as dt
import knn as knn
import gradient_boosting as gradientB

if __name__ == "__main__":
    arg1 = "Data/"  # Path of the data folder
    arg2 = True # Display caracteristics histograms
    arg3 = 0.85  # What is the max % of caracteristics similar in a 10% range with respect to the total range of the caracteristic
    arg4 = 8  # Number of K in KNN algorithm

    arg10 = 0.5
    arg11 = 100
    arg12 = 2
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
    
    kFold = 2
    lr = [0.1,0.5,0.75,1]
    estimator = [300,500,800]
    sample = [2]
    
    dataset = dt.Dataset(arg1, arg2, arg3)
    
    gd = gradientB.gradientBoosting(dataset, arg10, arg11, arg12) 
    gd.recherche_hyperparametres(kFold, lr, estimator, sample) 
    print(gd.entrainement()) 
    gd.run()
    