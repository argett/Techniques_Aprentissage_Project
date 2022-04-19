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
import matplotlib.pyplot as plt
import Dataset as dt
import random_forest as rForest
import knn as knn
import gradient_boosting as gradientB
import LinearDiscriminantAnalysis as lda
import NuSVC as Nusvc
import SVC as Svc
import sys


def Kcross_validation(kFold, model, model_knn, model_rdmForest, model_gradBoost, model_lda, model_svc, model_nusvc):
    if model == "knn":
        model_knn.recherche_hyperparametres(kFold)
        plt.plot(model_knn.err_train, label='Score train')
        plt.plot(model_knn.err_valid, label='Score valid')
        plt.legend()
        plt.title("KNN : Bonne réponse moyenne sur K-fold validation")
        plt.show()

    elif model == "rforest":
        model_rdmForest.recherche_hyperparametres(kFold)
        plt.plot(model_rdmForest.err_train, label='Score train')
        plt.plot(model_rdmForest.err_valid, label='Score valid')
        plt.legend()
        plt.title("Random Forest : Bonne réponse moyenne sur K-fold validation")
        plt.show()

    elif model == "gboost":
        model_gradBoost.recherche_hyperparametres(kFold)
        plt.plot(model_gradBoost.err_train, label='Score train')
        plt.plot(model_gradBoost.err_valid, label='Score valid')
        plt.legend()
        plt.title("Gradient Boosting : Bonne réponse moyenne sur K-fold validation avec distance de Manhattan")
        plt.show()

    elif model == "lda":
        model_lda.recherche_hyperparametres(kFold)
        plt.plot(model_lda.err_train, label='Score train')
        plt.plot(model_lda.err_valid, label='Score valid')
        plt.legend()
        plt.title("Linear Discriminant Analysis : Bonne réponse moyenne sur K-fold validation")
        plt.show()

    elif model == "svc":
        model_svc.recherche_hyperparametres(kFold)
        plt.plot(model_svc.err_train, label='Score train')
        plt.plot(model_svc.err_valid, label='Score valid')
        plt.legend()
        plt.title("Support Vector Classification : Bonne réponse moyenne sur K-fold validation")
        plt.show()

    elif model == "nusvc":
        model_nusvc.recherche_hyperparametres(kFold)
        plt.plot(model_lda.err_train, label='Score train')
        plt.plot(model_lda.err_valid, label='Score valid')
        plt.legend()
        plt.title("Nu Support Vector Classification : Bonne réponse moyenne sur K-fold validation")
        plt.show()

    else:
        errorParameters("<!> Model unknown <!>")


def train_test(dataset, model, model_knn, model_rdmForest, model_gradBoost, model_lda, model_svc, model_nusvc):
    if model == "knn":
        model_knn.entrainement(dataset.xTrain(), dataset.yTrain())
        print("score Test = " + str(round(model_knn.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_knn.err_train[-1], 4)*100) + "%")
        # res1 = model_knn.run()

    elif model == "rforest":
        model_rdmForest.entrainement(dataset.xTrain(), dataset.yTrain())
        print("score Test = " + str(round(model_rdmForest.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_rdmForest.err_train[-1], 4)*100) + "%")
        # res2 = model_rdmForest.run()

    elif model == "gboost":
        model_gradBoost.entrainement(dataset.xTrain(), dataset.yTrain())
        print("score Test = " + str(round(model_gradBoost.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_gradBoost.err_train[-1], 4)*100) + "%")
        # res3 = model_gradBoost.run()

    elif model == "lda":
        model_lda.entrainement(dataset.xTrain(), dataset.yTrain())
        print("score Test = " + str(round(model_lda.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_lda.err_train[-1], 4)*100) + "%")
        # res4 = model_lda.run()

    elif model == "svc":
        model_svc.entrainement(dataset.xTrain(), dataset.yTrain())
        print("score Test = " + str(round(model_svc.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_svc.err_train[-1], 4)*100) + "%")
        # res5 = model_svc.run()

    elif model == "nusvc":
        model_nusvc.entrainement(dataset.xTrain(), dataset.yTrain())
        print("score Test = " + str(round(model_nusvc.score(dataset.xTest(), dataset.yTest()), 4)*100) + "%, score train = " + str(round(model_nusvc.err_train[-1], 4)*100) + "%")
        # res6 = model_nusvc.run()

    else:
        errorParameters("<!> Model unknown <!>")


def errorParameters(message):
    print("Usage: python main.py ... and add the parameters\n")
    print("There are 4 mandatory parameters : \n")

    print("\t 1) Path: The path to the Data folder, string")
    print("\t 2) Display caracteristic (0 or 1), to display or not the selected histograms")
    print("\t 3) Feature selection: The maximum % of carecteristics in a 10% range of total caracteristics values [0.3,1], float \
         \n\t\t 1 = select all data, 0.85 = do not take every caracteristic where 85% of the data is in the 10% range of the total range")
    print("\t 4) Number of K for Train/Test split kcross-validation. min 3 or -1 for no kcross validation, int")
    print("\t 5) Allow the hyperparameter research (0 or 1): 0 = take the default or selected values, 1 = apply the hyperparameter research")
    print("\t 5bis) ! Only if you allowed the hyperparameter research ! Number of K in the kcross validation: [1,10] (10 can be long to compute for gradient boosting for example), integer")

    print("\n\nThe following is going to depend on the type of model you want :")

    print("\n\t\t --- KNN model ---")
    print("\t 6) knn")
    print("\t 7) Number of neighbour for KNN: The number of neighbour in the KNN-calssifyer. Minimum 1, integer")
    print("\t 8) Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. Minimum 1, integer")
    print("\t 9) To use the Manhattan distance or Euclidean. 1 = Manhattan, 0 = Euclidean, integer")

    print("\n\t\t --- Random Forest model ---")
    print("\t 6) rforest")
    print("\t 7) Number of threes in the random forest. [10,1000], integer")
    print("\t 8) Maximum depth of the tress. Minimum 1, integer")
    print("\t 9) Number of minimum samples to create a new node. Minimum 1, integer")
    print("\t 10) The number of features to consider when looking for the best split. Minimum 1, integer")
    print("\t 11) To use the Gini or the Entropy of the leaf quality mesure. \"gini\" or \"entropy\"")

    print("\n\t\t --- Gradient Boosting model ---")
    print("\t 6) gboost")
    print("\t 7) Learning rate. [0.0001,1], float")
    print("\t 8) The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance. [10,1000], integer")
    print("\t 9) The minimum number of samples required to split an internal node. [1,98], integer")

    print("\n\t\t --- LDA model ---")
    print("\t 6) lda")
    print("\t 7) solver: \"svd\" or \"lsqr\" or \"eigen\"")
    print("\t 8) shrinkage: [0,1], float")

    print("\n\t\t --- SVC model ---")
    print("\t 6) svc")
    print("\t 7) kernel: \"linear\", \"poly\", \"rbf\", \"sigmoid\"")
    print("\t 8) gamma: \"auto\" or \"scale\"")
    print("\t 9) C: [1,100000], int")

    print("\n\t\t --- NuSVC model ---")
    print("\t 6) nusvc")
    print("\t 7) kernel: \"linear\", \"poly\", \"rbf\", \"sigmoid\"")
    print("\t 8) gamma: \"auto\" or \"scale\"")

    print("\nExemple :")
    print("\tpython main.py Data 0 1 -1  0 knn")
    print("\tpython main.py Data 0 1 3 1 1 rforest 50/100 2/5/10 5/8/13 1/2 gini/entropy")
    print("\tpython main.py Data 0 1 -1 1 5 knn 1/2/3 10/50 0")
    print("\tpython main.py Data 0 1 3 1 5 svc 1/2/3 rbf/poly scale/auto")
    print("\tpython main.py Data 0 1 -1 0 nusvc\n")
    sys.exit(message)


if __name__ == "__main__":
    dataset = None
    model_knn = None
    model_rdmForest = None
    model_gradBoost = None
    model_lda = None
    model_svc = None
    model_nusvc = None

    arg1 = arg2 = arg3 = arg4 = arg5 = arg6 = arg7 = arg8 = arg9 = arg10 = None
    arg11 = arg12 = arg13 = arg14 = arg15 = arg16 = arg17 = arg18 = arg19 = None
    arg20 = arg21 = arg22 = arg23 = arg24 = arg25 = arg26 = arg27 = arg28 = None

    add = 0
    paramSize = len(sys.argv)

    if paramSize < 7:
        errorParameters("<!> Not enough paramaters <!>")
    else:
        arg1 = str(sys.argv[1])  # Path of the data folder
        arg2 = bool(int(sys.argv[2]))  # Display caracteristics histograms
        arg3 = float(sys.argv[3])  # What is the max % of caracteristics similar in a 10% range with respect to the total range of the caracteristic
        if arg3 < 0.1 or arg3 > 1:
            errorParameters("<!> Bad argument 3 <!>")

        arg4 = int(sys.argv[4])  # Number of K for Train/Test split kcross-validation. -1 for no kcross data validation
        arg5 = bool(int(sys.argv[5]))  # Boolean to allow the hyperparameters research or not
        if arg5:
            add = 1
            arg6 = int(sys.argv[6])  # Number of k in the k-cross validation

        arg7 = str(sys.argv[6 + add])  # The model of the algorithm

        # creation of the dataset
        dataset = dt.Dataset(arg1, arg2, arg3, arg4)
        if arg7 == "knn":
            if paramSize == 7:
                model_knn = knn.Knn(dataset)
            else:
                if paramSize != (10 + add):
                    errorParameters("<!> Not the correct number of parameters for the model <!>")

                if arg5:
                    # hyperparameters research
                    arg8 = str(sys.argv[7 + add])  # Number of K in KNN algorithm
                    arg9 = str(sys.argv[8 + add])  # Number of leaf size
                    arg10 = str(sys.argv[9 + add])  # 1=Using manhattan distance, 0=euclidean distance
                    # transform command line into iterrable lists
                    arg8 = list(map(int, list(arg8.split("/"))))
                    arg9 = list(map(int, list(arg9.split("/"))))
                    arg10 = list(map(bool, list(arg10.split("/"))))

                else:
                    arg8 = int(sys.argv[7 + add])  # Number of K in KNN algorithm
                    arg9 = int(sys.argv[8 + add])  # Number of leaf size
                    arg10 = bool(sys.argv[9 + add])  # 1=Using manhattan distance, 0=euclidean distance

                model_knn = knn.Knn(dataset, arg8, arg9, manhattan=(arg10))
                if arg5:  # to not have to resst each time we make a cross validation Train/Test
                    model_knn.addListParameters(arg8)
                    model_knn.addListParameters(arg9)
                    model_knn.addListParameters(arg10)

        elif arg7 == "rforest":
            if paramSize == 7:
                model_rdmForest = rForest.RandomForest(dataset)
            else:
                if paramSize != (12 + add):
                    errorParameters("<!> Not the correct number of parameters for the model <!>")

                if arg5:
                    # hyperparameters research
                    arg11 = str(sys.argv[7 + add])  # Number of threes in the random forest
                    arg12 = str(sys.argv[8 + add])  # Maximum depth of the trees
                    arg13 = str(sys.argv[9 + add])  # Number of minimum samples to create a new node
                    arg14 = str(sys.argv[10 + add])  # The number of features to consider when looking for the best split 1
                    arg15 = str(sys.argv[11 + add])  # The function to measure the quality of a split {"gini", "entropy"}
                    # transform command line into iterrable lists
                    arg11 = list(map(int, list(arg11.split("/"))))
                    arg12 = list(map(int, list(arg12.split("/"))))
                    arg13 = list(map(int, list(arg13.split("/"))))
                    arg14 = list(map(int, list(arg14.split("/"))))
                    arg15 = list(map(str, list(arg15.split("/"))))
                else:
                    arg11 = int(sys.argv[7 + add])  # Number of threes in the random forest
                    arg12 = int(sys.argv[8 + add])  # Maximum depth of the trees
                    arg13 = int(sys.argv[9 + add])  # Number of minimum samples to create a new node
                    arg14 = int(sys.argv[10 + add])  # The number of features to consider when looking for the best split 1
                    arg15 = str(sys.argv[11 + add])  # The function to measure the quality of a split {"gini", "entropy"}

                model_rdmForest = rForest.RandomForest(dataset, arg11, arg12, arg13, arg14, arg15)
                if arg5:  # to not have to resst each time we make a cross validation Train/Test
                    model_rdmForest.addListParameters(arg11)
                    model_rdmForest.addListParameters(arg12)
                    model_rdmForest.addListParameters(arg13)
                    model_rdmForest.addListParameters(arg14)
                    model_rdmForest.addListParameters(arg15)

        elif arg7 == "gboost":
            if paramSize == 7:
                model_gradBoost = gradientB.GradientBoosting(dataset)
            else:
                if paramSize != (10 + add):
                    errorParameters("<!> Not the correct number of parameters for the model <!>")
                if arg5:
                    # hyperparameters research
                    arg16 = str(sys.argv[7 + add])  # Learning rate
                    arg17 = str(sys.argv[8 + add])  # Number of estimator
                    arg18 = str(sys.argv[9 + add])  # Minimum number of sample to create a new node
                    # transform command line into iterrable lists
                    arg16 = list(map(int, list(arg16.split("/"))))
                    arg17 = list(map(int, list(arg17.split("/"))))
                    arg18 = list(map(int, list(arg18.split("/"))))
                else:
                    arg16 = int(sys.argv[7 + add])  # Learning rate
                    arg17 = int(sys.argv[8 + add])  # Number of estimator
                    arg18 = int(sys.argv[9 + add])  # Minimum number of sample to create a new node

                model_gradBoost = gradientB.GradientBoosting(dataset, arg16, arg17, arg18)
                if arg5:  # to not have to resst each time we make a cross validation Train/Test
                    model_gradBoost.addListParameters(arg8)
                    model_gradBoost.addListParameters(arg9)
                    model_gradBoost.addListParameters(arg10)

        elif arg7 == "lda":
            if len(sys.argv) == 7:
                model_lda = lda.LDA(dataset)
            else:
                if paramSize != (9 + add):
                    errorParameters("<!> Not the correct number of parameters for the model <!>")
                if arg5:
                    # hyperparameters research
                    arg19 = str(sys.argv[7 + add])  # Learning rate
                    arg20 = str(sys.argv[8 + add])  # Number of estimator
                    # transform command line into iterrable lists
                    arg19 = list(map(str, list(arg19.split("/"))))
                    arg20 = list(map(float, list(arg20.split("/"))))
                else:
                    arg19 = str(sys.argv[7 + add])  # Learning rate
                    arg20 = float(sys.argv[8 + add])  # Number of estimator

                model_lda = lda.LDA(dataset, arg19, arg20)
                if arg5:  # to not have to resst each time we make a cross validation Train/Test
                    model_lda.addListParameters(arg19)
                    model_lda.addListParameters(arg20)

        elif arg7 == "svc":
            if len(sys.argv) == 7:
                model_svc = Svc.svc(dataset)
            else:
                if paramSize != (10 + add):
                    errorParameters("<!> Not the correct number of parameters for the model <!>")
                if arg5:
                    # hyperparameters research
                    arg21 = str(sys.argv[7 + add])  # Learning rate
                    arg22 = str(sys.argv[8 + add])  # Number of estimator
                    arg23 = str(sys.argv[9 + add])  # Number of estimator
                    # transform command line into iterrable lists
                    arg21 = list(map(int, list(arg21.split("/"))))
                    arg22 = list(map(str, list(arg22.split("/"))))
                    arg23 = list(map(str, list(arg23.split("/"))))
                else:
                    arg21 = int(sys.argv[7 + add])  # Learning rate
                    arg22 = str(sys.argv[8 + add])  # Number of estimator
                    arg23 = str(sys.argv[9 + add])  # Number of estimator

                model_svc = Svc.svc(dataset, arg21, arg22, arg23)
                if arg5:  # to not have to resst each time we make a cross validation Train/Test
                    model_svc.addListParameters(arg21)
                    model_svc.addListParameters(arg22)
                    model_svc.addListParameters(arg23)

        elif arg7 == "nusvc":
            if len(sys.argv) == 7:
                model_nusvc = Nusvc.NUSVC(dataset)
            else:
                if paramSize != (9 + add):
                    errorParameters("<!> Not the correct number of parameters for the model <!>")
                if arg5:
                    # hyperparameters research
                    arg24 = str(sys.argv[7 + add])  # Learning rate
                    arg25 = str(sys.argv[8 + add])  # Number of estimator
                    # transform command line into iterrable lists
                    arg24 = list(map(str, list(arg24.split("/"))))
                    arg25 = list(map(str, list(arg25.split("/"))))
                else:
                    arg24 = str(sys.argv[7 + add])  # Learning rate
                    arg25 = str(sys.argv[8 + add])  # Number of estimator

                model_musvc = Nusvc.NUSVC(dataset, arg24, arg25)
                if arg5:  # to not have to resst each time we make a cross validation Train/Test
                    model_nusvc.addListParameters(arg24)
                    model_nusvc.addListParameters(arg25)
        else:
            errorParameters("<!> Model unknown <!>")

    if arg5:  # Cross validation
        if(dataset.Kcross):  # dataset kcross
            for ki in range(dataset.split):
                print("================= Recherche hyperparamètres, kcross de dataset split = " + str(ki) + " =================")
                dataset.split_data(ki)
                Kcross_validation(arg6, arg7, model_knn, model_rdmForest, model_gradBoost, model_lda, model_svc, model_nusvc)
        else:
            Kcross_validation(arg6, arg7, model_knn, model_rdmForest, model_gradBoost, model_lda, model_svc, model_nusvc)

    if(dataset.Kcross):
        for ki in tqdm(range(dataset.split)):
            print("\n================= Tests kcross de dataset split pour entrainement = " + str(ki) + " =================\n")
            dataset.split_data(ki)
            train_test(dataset, arg7, model_knn, model_rdmForest, model_gradBoost, model_lda, model_svc, model_nusvc)
    else:
        resultats = train_test(dataset, arg7, model_knn, model_rdmForest, model_gradBoost, model_lda, model_svc, model_nusvc)
