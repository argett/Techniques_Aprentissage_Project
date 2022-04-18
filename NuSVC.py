# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:55:10 2022

@author: Tsiory
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.svm import NuSVC


class NUSVC():
    def __init__(self, dataHandler, proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        proportion : float, optional
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The
            default is 0.2.

        Returns
        -------
        None.

        """
        self.NSVC = None
        self.dh = dataHandler

        self.proportion = proportion

        self.probability = probability
        self.learning_rate = lr
        self.max_depth = max_depth
        self.max_features = max_features
        self.estimators = estimators
        self.min_sample = min_sample

        self.err_train = []
        self.err_valid = []
