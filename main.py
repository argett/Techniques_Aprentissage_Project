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

from tqdm import tqdm
# import matplotlib.pyplot as plt
import Dataset as dt
import random_forest as rForest
import knn as knn
import gradient_boosting as gradientB
import numpy as np
import LinearDiscriminantAnalysis as lda
import NuSVC as nusvc
import SVC as svc


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

        mostCommon_label = max(classes, key=classes.count)

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


def Kcross_validation(arg6, model_knn, model_rdmForest, model_gradBoost):
    kFold = arg6

    # KNN
    ks = [3] 
    ls = [2] 
    model_knn.recherche_hyperparametres(kFold, ks, ls)

    # Random Forest
    nb_trees = [350]
    maxDepth = [50]
    random_state = [50]
    max_features = [25]
    min_sample = [2]
    criterion = ["gini", "entropy"]
    model_rdmForest.recherche_hyperparametres(kFold, nb_trees, maxDepth, random_state, max_features, min_sample, criterion)

    """
    # Gradient Boosting
    lr = [0.1]
    estimator = [400]
    sample = [2]
    model_gradBoost.recherche_hyperparametres(kFold, lr, estimator, sample)
    """


def train_test(model_knn, model_rdmForest, model_gradBoost, model_lda, dataset):
    print("KNN :")
    model_knn.entrainement(dataset.xTrain(), dataset.yTrain())
    print("score Test = " + str(round(model_knn.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_knn.err_train[-1], 4)*100) + "%")
    # res1 = model_knn.run()

    print("Random forest :")
    model_rdmForest.entrainement(dataset.xTrain(), dataset.yTrain())
    print("score Test = " + str(round(model_rdmForest.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_rdmForest.err_train[-1], 4)*100) + "%")
    # res2 = model_rdmForest.run()

    print("Gradient boosting :")
    model_gradBoost.entrainement(dataset.xTrain(), dataset.yTrain())
    print("score Test = " + str(round(model_gradBoost.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_gradBoost.err_train[-1], 4)*100) + "%")
    # res3 = model_gradBoost.run()

    print("LDA :")
    model_lda.hyperparameters_search(dataset.xTrain(), dataset.yTrain())
    print("score Test = " + str(round(model_lda.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_lda.err_train[-1], 4)*100) + "%")

    return 0  # [res1.tolist(), res2.tolist(), res3.tolist()]


if __name__ == "__main__":
    arg1 = "Data/"  # Path of the data folder
    arg2 = False  # Display caracteristics histograms
    arg3 = 1  # What is the max % of caracteristics similar in a 10% range with respect to the total range of the caracteristic
    arg4 = 3  # Number of K for Train/Test split kcross-validation. -1 for no kcross validation

    arg5 = 0     # Boolean to allow the hyperparameters research or not
    arg6 = 1     # Number of k in the k-cross validation

    # KNN
    arg7 = 3  # Number of K in KNN algorithm
    arg8 = 2  # Number of leaf size
    arg9 = True  # 1=Using manhattan distance, 0=euclidean distance

    # random forest parameters
    arg10 = 350  # Number of threes in the random forest 50
    arg11 = 50  # maximum depth of the trees 10
    arg12 = 2  # Number of minimum samples to create a new node 2
    arg13 = 1  # The number of features to consider when looking for the best split 1
    arg14 = "gini"  # The function to measure the quality of a split {"gini", "entropy"}

    # Gradient Boosting
    arg15 = 0.1  # learning rate
    arg16 = 400  # number of estimator
    arg17 = 5  # Minimum number of sample to create a new node

    # LDA
    # ...

    """
    if len(sys.argv) < 8:
        print("Usage: python main.py ... and add the parameters\n")
        print("There are 4 mandatory parameters : \n")

        print("\t Path: The path to the Data folder, string")
        print("\t Display caracteristic (0 or 1), to display or not the selected histograms")
        print("\t Feature selection: The maximum % of carecteristics in a 10% range of total caracteristics values [0.3,1], float
             \n\t\t 1 = select all data, 0.85 = don't take every caracteristic where 85% of the data is in the 10% range of the total range")
        print("\t Number of K for Train/Test split kcross-validation. min 3 or -1 for no kcross validation, int")
        print("\t Allow the hyperparameter research (0 or 1): 0 = take the default or selected values, 1 = apply the hyperparameter research")
        print("\t -only if you allowed the hyperparameter research- Number of K in the kcross validation: [1,10] (10 can be long to compute for gradient boosting for example), integer")

        print("The following is going to depend on the type of model you want :")

        print("\n\t\t --- KNN model ---")
        print("\t Number of neighbour for KNN: The number of neighbour in the KNN-calssifyer. Minimum 1, integer")
        print("\t Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. Minimum 1, integer")
        print("\t To use the Manhattan distance or Euclidean. 1 = Manhattan, 0 = Euclidean, integer")

        print("\n\t\t --- Random Forest model ---")
        print("\t Number of threes in the random forest. [10,1000], integer")
        print("\t Maximum depth of the tress. Minimum 1, integer")
        print("\t Number of minimum samples to create a new node. Minimum 1, integer")
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
    model_knn = knn.knn(dataset, arg7, arg8, manhattan=(arg9))
    model_rdmForest = rForest.randomForest(dataset, arg10, arg11, arg12, arg13, arg14)
    model_gradBoost = gradientB.gradientBoosting(dataset, arg15, arg16, arg17)
    model_lda = lda.LinearDiscriminantAnalysis(dataset)

    if arg5 == 1:  # Cross validation
        if(dataset.Kcross):  # dataset kcross
            for ki in range(dataset.split):
                print("================= Recherche hyperparamètres, kcross de dataset split = " + str(ki) + " =================")
                dataset.split_data(ki)
                Kcross_validation(arg6, model_knn, model_rdmForest, model_gradBoost)
        else:
            Kcross_validation(arg6, model_knn, model_rdmForest, model_gradBoost)

    """
    plt.plot(model_knn.err_train, label='Score train')
    plt.plot(model_knn.err_valid, label='Score valid')
    plt.legend()
    plt.title("KNN : Bonne réponse moyenne sur K-fold validation avec distance de Manhattan")
    plt.show()
    """

    if(dataset.Kcross):
        for ki in tqdm(range(dataset.split)):
            print("\n================= Tests kcross de dataset split pour entrainement = " + str(ki) + " =================\n")
            dataset.split_data(ki)
            train_test(model_knn, model_rdmForest, model_gradBoost, model_lda, dataset)
    else:
        resultats = train_test(model_knn, model_rdmForest, model_gradBoost, model_lda, dataset)
        processResult(resultats)
