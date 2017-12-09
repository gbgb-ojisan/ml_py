#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:25:04 2017

@author: gbgb
"""
#%% import
import mglearn
import matplotlib.pyplot as plt
import numpy as np

#%% NB classifier

X = np.array([[0,1,0,1],
             [1,0,1,1],
             [0,0,0,1],
             [1,0,1,1]]) # 4 dataset with 4 features
y = np.array([0,1,0,1]) # Class 0 and 1

counts = {}
for label in np.unique(y): # np.unique returns 0,1
    # loop for each classes
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))

#%% Decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set : {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set : {:.3f}".format(tree.score(X_test, y_test)))

#%% Analysis of decision tree (export dot file)

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

#%% Analysis of decision tree (import dot file and show graph)

import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)