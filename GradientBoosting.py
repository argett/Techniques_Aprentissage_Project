# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:21:33 2022

@author: Lilian
"""
from GeneralModel import CommonModel
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm


class GradientBoosting(CommonModel):
    def __init__(self, dataHandler, lr=0.1, estimators=350, min_sample=2, proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        lr : float, optional
            The learning rate corresponding to the gradient descent.
            Default = 0.1
        estimators : int, optional
            The number of boosting stages to perform. A large number usually
            results in better performance. Default = 350.
        min_sample : int, optional
            The minimum number of samples required to split an internal node.
            Default = XXX.
        proportion : float, optional
            Proportion of learn/verify data splitting 0.6 < x < 0.9.
            Default = 0.2.

        Returns
        -------
        None.

        """
        CommonModel.__init__(self, dataHandler, proportion)
        self.gradientBoosting = None

        self.learning_rate = lr
        self.estimators = estimators
        self.min_sample = min_sample

    def recherche_hyperparametres(self, num_fold):
        """
        The function is going to try every possibility of combinaison within
        the given lists of parameters to find the one which has the less error
        on the model.

        Parameters
        ----------
        num_fold : int
            The number of time the the k-cross validation is going to be made.
        learning_rate : list[float]
            List of all learning rate shrinks the contribution of each tree by
            learning_rate to try. There is a trade-off between learning_rate
            and n_estimators.
        n_estimators : list[int]
            List of all number of boosting stages to perform. Gradient
            boosting is fairly robust to over-fitting so a large number
            usually results in better performance..
        min_samples_split : list[int]
            List of all minimum numbers of samples required to split an
            internal node to try.

        Returns
        -------
        None.
        """
        liste_res = []
        liste_lr = []
        liste_est = []
        liste_sam = []

        better = False
        meilleur_lr = None
        meilleur_estimantor = None
        meilleur_sample = None

        # On teste plusieurs degr??s du polyn??me
        for lr in tqdm(self.getListParameters(0)):
            for esti in self.getListParameters(1):
                for samp in self.getListParameters(2):
                    sum_result = 0

                    self.learning_rate = lr
                    self.estimators = esti
                    self.min_sample = samp

                    for k in range(num_fold):  # K-fold validation
                        X_learn, X_verify, y_learn, y_verify = train_test_split(self.dh.xTrain(), self.dh.yTrain(), test_size=self.proportion, random_state=k, shuffle=True)
                        self.entrainement(X_learn, y_learn)
                        self.err_valid.append(self.score(X_verify, y_verify))
                        sum_result += self.err_valid[-1]

                    # On regarde la moyenne des erreurs sur le K-fold
                    avg_res_locale = sum_result/(num_fold)

                    liste_res.append(avg_res_locale)
                    liste_lr.append(lr)
                    liste_est.append(esti)
                    liste_sam.append(samp)

                    if(avg_res_locale > self.betterValidationScore):
                        better = True
                        self.betterValidationScore = avg_res_locale
                        meilleur_lr = lr
                        meilleur_estimantor = esti
                        meilleur_sample = samp

        if better:
            self.learning_rate = meilleur_lr
            self.estimators = meilleur_estimantor
            self.min_sample = meilleur_sample

            print("\nLes meilleurs parametres parmis ceux essayes sont :")
            print("\tMeilleur learning rate = " + str(meilleur_lr))
            print("\tMeilleur nombre d'estimators = " + str(meilleur_estimantor))
            print("\tMeilleur sample= " + str(meilleur_sample))

        plt.plot(liste_res)
        plt.title("Gradient boosting : Bonne r??ponse moyenne sur les K validations")
        plt.show()
        plt.plot(liste_lr)
        plt.title("Gradient boosting : Valeurs du taux d'apprentissage")
        plt.show()
        plt.plot(liste_est)
        plt.title("Gradient boosting : Valeurs du nombre d'estimators")
        plt.show()
        plt.plot(liste_sam)
        plt.title("Gradient boosting : Valeurs du nombre de sample")
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
        self.gradientBoosting = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.estimators, min_samples_split=self.min_sample)
        self.gradientBoosting.fit(xData, yData.ravel())
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
        return self.gradientBoosting.score(xData, yData)

    def run(self):
        """
        Run the model on pre-loaded testing data.

        Returns
        -------
        List[string]
            Every class or label deduced from the entry dataset with the
            trained model
        """
        return self.gradientBoosting.predict(self.dh.xUnknownData())
