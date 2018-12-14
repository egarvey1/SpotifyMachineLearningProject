# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:16:34 2018

@author: PC
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score as cv_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from Data_Extract import data_loader
from SPOTIPY_CONSTANTS import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 
from sklearn import metrics


class ML_random_forest():
    def __init__(self, dataset, num_trees, depth, sample_split, samples_leaf, weight, max_features, node, column_drops): #, column_drops
        self.dataset = dataset
        self.num_trees = num_trees
        self.depth = depth
        self.sample_split = sample_split
        self.samples_leaf = samples_leaf
        self.weight = weight
        self.max_features = max_features
        self.node = node
        self.column_drops = column_drops
#        self.kfold_splits=splits
#        self.leaf_samples = leaf_samples
#        self.tree_depth = tree_depth
#        self.max_features = max_features
        self.testing_features=False
        
        
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
        self.clf = None
        self.xval_score = None
        self.matrix = None
        self.report = None
        
        self.f_score = None
        
        
        if self.testing_features==False:
            self.assign_tree_data()
    #        self.encode_data()
            self.separate_data()
            self.create_random_forest()
            self.cross_validate()
        else: 
            self.feature_testing()
        
    def feature_testing(self):
        for feature in attributes_all:
            
            self.data = self.dataset.drop(feature, axis=1)
            self.assign_tree_data()
            self.separate_data()
            self.create_random_forest()
            self.cross_validate()
            print("Score after dropping "+feature+ " : "+str(self.f_score))
    
    def assign_tree_data(self):
        
        drop_cols = unused_cols+self.column_drops
        self.data = self.dataset.drop(drop_cols, axis=1)
#        self.data = self.dataset.drop(unused_cols, axis=1)
        self.target= self.dataset.rating.values
       
        
    def encode_data(self):
        
        labelEncode = preprocessing.LabelEncoder()
        integer_encoded = labelEncode.fit_transform(self.target)
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        self.target = onehot_encoded
        
    def separate_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data,
                                                    self.target,
                                                    test_size=0.30)

    def create_random_forest(self):
        self.clf = RandomForestClassifier(n_estimators = self.num_trees, 
                                          random_state = 42, 
                                          max_depth = self.depth, 
                                          min_samples_split = self.sample_split , 
                                          min_samples_leaf = self.samples_leaf, 
                                          min_weight_fraction_leaf = self.weight, 
                                          max_features = self.max_features, 
                                          max_leaf_nodes = self.node )
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        
        xtrain =self.X_train
        xtest =self.X_test
        ytrain =self.y_train 
        ytest =self.y_test 
        ypred=self.y_pred 
        target = self.target
      
        
    def cross_validate(self):
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        self.report = classification_report(self.y_test, self.y_pred)
        
        self.f_score = f1_score(self.y_test, self.y_pred, average="weighted")
        ##Entropy in decision tree as a good way to evaluate significance of 
  
      
if __name__ =="__main__":
    
    afscore=[]
    efscore=[]

    n_estimators = range(10, 1000, 50)    
#    criteria = ["entropy", "gini"]
    max_depth = range(2,10)
    min_samples_split = np.linspace(0.001,0.20, 10)
    min_samples_leaf = np.linspace(0.001,0.1, 10)
    min_weight_fraction_leaf = np.linspace(0,0.04, 10)
    max_features = np.linspace(0.001,1.0, 10)
    max_leaf_nodes = range(5, 50, 2)
#    min_impurity_decrease = 
    

    
    
#    for trees in n_estimators:
#        print(trees)
#
#    for sample in min_samples_split:
#    for sample_split in min_samples_split:
#    for node in max_leaf_nodes:
##    for depth in max_depth:
#        print(node)
        
    alice= data_loader(ALICE_USER_ID, ALICE_CLIENT_ID, ALICE_CLIENT_SECRET,
                                                           ALICE_PLAYLISTS, 8)
    emma = data_loader(EMMA_USER_ID, EMMA_CLIENT_ID, EMMA_CLIENT_SECRET,
                                                           EMMA_PLAYLISTS, 6)
    alice_data = alice.playlist_dataset
    emma_data = emma.playlist_dataset
    print("Alice")
    alice_forest = ML_random_forest(alice_data, num_trees = 300,depth = 5,
                                    sample_split = 0.023111111111111114, 
                                    samples_leaf = 0.01188888888888889 , 
                                    weight = 0.017777777777777778, 
                                    max_features = 0.223, 
                                    node = 19,
                                    column_drops = ALICE_DROPS) 
    print("Emma")
    emma_forest = ML_random_forest(emma_data, num_trees =300  ,depth = 6,
                                   sample_split = 0.045222222222222226, 
                                   samples_leaf = 0.0064444444444444445, 
                                   weight = 0.0044444444444444444, 
                                   max_features = 0.667, 
                                   node = 20, 
                                   column_drops = EMMA_DROPS)   #20    
    afscore.append(alice_forest.f_score)
    efscore.append(emma_forest.f_score)


    print("Alice: ")
    print(alice_forest.report)

    print("Emma: ")
    print(emma_forest.report)
    
#    plt.plot(max_leaf_nodes, afscore, label = "Alice's Data")
#    plt.plot(max_leaf_nodes, efscore, label = "Emma's Data")
#    plt.legend()
#    plt.title("Effect of Maximum Leaf Nodes in the Forest")
#    plt.xlabel("Maximum Leaf Nodes in the Forest")
#    plt.ylabel("Accuracy (%)")
    
        
       