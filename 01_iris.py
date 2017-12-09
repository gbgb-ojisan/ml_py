#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 23:00:13 2017

@author: gbgb
"""
#%%load
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#%% disp keys
print("Key of iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:] + "\n...")
print("{}".format(iris_dataset['data'][:5]))

#%% split test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#%% Inspection
import pandas as pd
from IPython.display import display 
# X_train -> DataFrame
# Named columns using feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# scatter
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                        hist_kwds={'bins':20}, s=60, alpha=.8, cmap='jet')

#%% Create the model (k-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# it returns
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=1, n_neighbors=1, p=2,
#                     weights='uniform')

#%% input the new iris
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

#%% predict
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
        iris_dataset['target_names'][prediction]))

#%% Accuracy
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))