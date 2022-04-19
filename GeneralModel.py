# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:19:20 2022

@author: Lilian
"""

class CommonModel():
    def __init__(self, dataHandler, proportion):
        self.dh = dataHandler    
        self.proportion = proportion  
        
        self.err_train = []
        self.err_valid = []
        
        self.betterValidationScore = 0
