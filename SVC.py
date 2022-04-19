# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:56:05 2022

@author: Tsiory
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC


class svc():
    def __init__(self, dataHandler, C=1.0, kernel='rbf', gamma='scale', shrinking=True, proportion=0.2):
        """
        Create an instance of the class

        Parameters
        ----------
        dataHandler : Dataset
            The dataset with everything loaded from the main
        C : float, optional
            Regularization parameter. Defaults to 1.0.
        kernel : str
            Specifies the kernel type to be used in the algorithm. Defaults to
            'rbf'.
        gamma : str
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
            Defaults to 'scale'.
        shrinking : bool
            Whether to use the shrinking heuristic. Defaults to True.
        proportion : float
            Proportion of learn/verify data splitting 0.05 < x < 0.30. The
            default is 0.2.

        Returns
        -------
        None.

        """
        # C, kernel, gamma, shrinking
        self.Svc = None
        self.dh = dataHandler

        self.proportion = proportion

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.shrinking = shrinking

        self.err_train = []
        self.err_valid = []
