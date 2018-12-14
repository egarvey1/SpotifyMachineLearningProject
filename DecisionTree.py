# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:48:31 2018

@author: Emma
"""


import numpy as np
from sklearn.model_selection import cross_val_score as cv_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from Data_Extract import data_loader
from SPOTIPY_CONSTANTS import *
import matplotlib.pyplot as plt



class ML_decision_tree():
    def __init__(self, dataset, splits, tree_depth, leaf_samples, max_features, column_drops):
        self.dataset = dataset
        self.kfold_splits=splits
        self.leaf_samples = leaf_samples
        self.tree_depth = tree_depth
        self.max_features = max_features
        self.column_drops = column_drops
        
        self.testing_features = False
        
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
            self.separate_data()
            self.create_decision_tree()
            self.cross_validate()
        else: 
            self.feature_testing()
        
        
        
    def feature_testing(self):
        for feature in attributes_all:
            
            self.data = self.dataset.drop(feature, axis=1)
            self.assign_tree_data()
            self.separate_data()
            self.create_decision_tree()
            self.cross_validate()
            print("Score after dropping "+feature+ " : "+str(self.f_score))
        
    def assign_tree_data(self):
        drop_cols = unused_cols+self.column_drops
        self.data = self.dataset.drop(drop_cols, axis=1)
#        self.data = self.dataset.drop(self.column_drops, axis=1)
        data = self.data
        self.target= self.dataset.rating
       
        
        
        
    def separate_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data,
                                                    self.target,
                                                    test_size=0.20)
       
        
        
    def create_decision_tree(self):
        self.clf = DecisionTreeClassifier(random_state=0,max_depth = 
                                          self.tree_depth, 
                                          min_samples_leaf=self.leaf_samples, 
                                          max_features = self.max_features)
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        
      
        
    def cross_validate(self):
        self.xval_score = cv_score(self.clf,self.data, self.target,
                                                   cv=self.kfold_splits )
        self.matrix = confusion_matrix(self.y_test, self.y_pred)
        self.report = classification_report(self.y_test, self.y_pred)
        
        self.f_score = f1_score(self.y_test, self.y_pred, average="weighted")
        
        
        
        
        
      



if __name__ =="__main__":
    
    afscore=[]
    efscore=[]
    alice= data_loader(ALICE_USER_ID, ALICE_CLIENT_ID, ALICE_CLIENT_SECRET,
                                                           ALICE_PLAYLISTS, 8)
    emma = data_loader(EMMA_USER_ID, EMMA_CLIENT_ID, EMMA_CLIENT_SECRET,
                                                           EMMA_PLAYLISTS, 6)
    alice_data = alice.playlist_dataset
    emma_data = emma.playlist_dataset
    

    alice_tree = ML_decision_tree(alice_data, splits =36,tree_depth = 6,
                                     leaf_samples=  .05,max_features= 8, column_drops = ALICE_DROPS)

    emma_tree = ML_decision_tree(emma_data, splits = 30, tree_depth = 6,
                                     leaf_samples=  .05,max_features= 8, column_drops = EMMA_DROPS)    


    print("Alice: ")
    print(alice_tree.report)

    print("Emma: ")
    print(emma_tree.report)
    

    

 
        









