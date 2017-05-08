#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:10:52 2017

@author: sezan92
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier

#knn with gridsearch
def ClassifierSelect(X,y,num_labels=2,SVMFlag=True):
    
    print "Knn Training..."
    knn = KNeighborsClassifier()
    k_range = list(range(1,31))
    leaf_range = list(range(1,40))
    weight_options = ['uniform', 'distance']
    algorithm_options =  ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_gridKnn = dict(n_neighbors = k_range,
                         weights = weight_options,
                         algorithm = algorithm_options
                         #leaf_size = leaf_range
                         )
    gridKNN = GridSearchCV(knn,param_gridKnn,cv=10,
                           scoring = 'accuracy') 
    gridKNN.fit(X,y)
    
    print "Knn Score "+ str(gridKNN.best_score_)
    print "Knn  best Params "+str(gridKNN.best_params_)
    Best = gridKNN
    BestScore = gridKNN.best_score_
    #LogReg with gridSearch
    
    print "Logistic Regression Training..."
    logreg = LogisticRegression()
    penalty_options =['l1','l2']
    solver_options = ['liblinear','newton_cg','lbfgs','sag']
    tol_options = [0.0001,0.00001,0.000001,0.000001]
    param_gridLog = dict(penalty=penalty_options,
                         tol=tol_options)
    gridLog = GridSearchCV(logreg,param_gridLog,cv=10,scoring='accuracy')
    gridLog.fit(X,y)
    
    
    print "LogReg Score "+ str(gridLog.best_score_)
    print "LogReg  best Params "+str(gridLog.best_params_)
    
    if gridLog.best_score_ > BestScore:
        Best = gridLog
        BestScore= gridLog.best_score_
    
    #NN with gridSearch
    
    print "Neural Network Training...."
    FirstLayer = (X.shape[1]+num_labels)/2
    SecondLayer = (FirstLayer+num_labels)/2
    ThirdLayer = (SecondLayer+num_labels)/2
    NN = MLPClassifier(hidden_layer_sizes=  (FirstLayer,SecondLayer,ThirdLayer))
    activation_options = ['identity', 'logistic', 'tanh', 'relu']
    solver_options =['lbfgs', 'sgd', 'adam']
    learning_rate_options = ['constant', 'invscaling', 'adaptive']
    param_gridNN = dict(activation=activation_options,
                        solver=solver_options,
                        learning_rate = learning_rate_options)
    gridNN = GridSearchCV(NN,param_gridNN,cv=10,
                          scoring = 'accuracy')
    gridNN.fit(X,y)
    
    if gridNN.best_score_>BestScore:
        Best=gridNN
        BestScore = gridNN.best_score_
    
    print "NN Score "+ str(gridNN.best_score_)
    print "NN  best Params "+str(gridNN.best_params_)
    
    #SVM with SVC
    
    
    
    if SVMFlag is True:
        print "SVM training. Caution It is slowest to train...."
        svmNu = NuSVC()
        nu_options =np.arange(0.1,1,0.1)
        kernel_options = [ 'linear', 'sigmoid', 'rbf']
        
        param_gridSVMNu = dict(kernel = kernel_options,nu =
                               nu_options)
        
        gridSVMNu = GridSearchCV(svmNu,param_gridSVMNu,cv=10,
                                 scoring = 'accuracy')
        gridSVMNu.fit(X,y)
        print "SVM with NuSVC Score "+str(gridSVMNu.best_score_)
        print "SVM with NuSVC best Params"+str(gridSVMNu.best_params_)
     
        if gridSVMNu.best_score_>BestScore:
            Best = gridSVMNu
            BestScore =gridSVMNu.best_score_
        #Random Forest
    print "DTree Training ..."
    dtree = DecisionTreeClassifier(random_state=0)
    criterion_options = ['gini','entropy']
    splitter_options =['best','random']
    
    param_gridDtree = dict(criterion =criterion_options,splitter=splitter_options)
    
    gridDtree = GridSearchCV(dtree,param_gridDtree,cv=10,scoring='accuracy')
    gridDtree.fit(X,y)
    
    
    print "Decision Tree Score "+str(gridDtree.best_score_)
    print "Decision Tree params "+str(gridDtree.best_params_)
    
    if gridDtree.best_score_>BestScore:
        Best = gridDtree
        BestScore = gridDtree.best_score_
    
    #Random Forest Classifier with GridSearch
    print "Randomforest Training ...."
    random = RandomForestClassifier()
    n_estimators_range = list(range(1,31))
    criterion_options = ['gini','entropy']
    max_features_options =['auto','log2', None]
    param_grid = dict(n_estimators =n_estimators_range,
                      criterion= criterion_options,
                      max_features =max_features_options)
    gridRandom = GridSearchCV(random,param_grid,cv=10,
                              scoring='accuracy')
    gridRandom.fit(X,y)
    
    if gridRandom.best_score_>BestScore:
        Best = gridRandom
        BestScore = gridRandom.best_score_
    
    print "RTrees Score "+str(gridRandom.best_score_)
    print "RTrees Best Params " +str(gridRandom.best_params_)
    
    return Best
