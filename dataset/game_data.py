# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 01:38:49 2020

@author: Abhinav
"""

import numpy as np
import labels

def load_data():
    X = np.load('C:\\Users\\Abhinav\\Documents\\MACS\\Machine Learning\\personal projects\\Reinforcement Learning\\Duck Hunt AI\\DuckHunt-Python-master\\AI\\dataset\\X.npy')
    Y = np.load('C:\\Users\\Abhinav\\Documents\\MACS\\Machine Learning\\personal projects\\Reinforcement Learning\\Duck Hunt AI\\DuckHunt-Python-master\\AI\\dataset\\Y.npy')
    X_train = X.copy()
    Y_train = Y.copy()
    X_test = []
    Y_test = []
    black_test_count = 0
    red_test_count = 0
    blue_test_count = 0
    object_test_count = 0
    indices = []
    for i in range(len(Y)):
        #Black duck samples for test
        if(Y[i] == labels.BLACK and black_test_count < 3):
            num = np.random.randint(0, 3)
            if(num == 1):
                X_test.append(X[i])
                Y_test.append(Y[i])
                indices.append(i)
                black_test_count += 1
        
        #Red duck samples for test
        if(Y[i] == labels.RED and red_test_count < 3):
            num = np.random.randint(0, 3)
            if(num == 1):
                X_test.append(X[i])
                Y_test.append(Y[i])
                indices.append(i)
                red_test_count += 1
        
        #Blue duck samples for test
        if(Y[i] == labels.BLUE and blue_test_count < 3):
            num = np.random.randint(0, 3)
            if(num == 1):
                X_test.append(X[i])
                Y_test.append(Y[i])
                indices.append(i)
                blue_test_count += 1
                
        #Other object samples for test
        if(Y[i] == labels.OBJECT and object_test_count < 2):
            num = np.random.randint(0, 2)
            if(num == 1):
                X_test.append(X[i])
                Y_test.append(Y[i])
                indices.append(i)
                object_test_count += 1
                
    
    X_train = np.delete(X_train, indices, 0)
    Y_train = np.delete(Y_train, indices, 0)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # print the array
    print('Test samples: ', len(X_test))
    print('Train samples:', len(X_train))
    return ((X_train, Y_train), (X_test, Y_test))
