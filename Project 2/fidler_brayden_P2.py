# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:43:32 2020

@author: bfidler
"""
import numpy as np
import matplotlib.pyplot as plt

def costFunction(w, X, y):
    
    #H is m*1 vector of hypothesis function results
    H = 1 / (1 + np.exp(-np.dot(X,w))) 
    
    #Inserted error for perfect values
    error = .00000000001
    
    #Eleminating Perfect 1s and 0s from H
    H = np.where(H == 0, H + error, H)
    H = np.where(H == 1, H - error, H)
    
    #Cost is m*1 vector of cost incurred for each row of data
    Cost = -(y * np.log(H)) - (1 - y) * np.log(1 - H)
    
    #Returning single number as J function
    return (np.sum(Cost) / y.shape[0])


def gradientDescent(X, y, alpha=0.1, iters=100):
    
    #Getting rows,cols and random initiak weights
    m = y.shape[0]
    n = X.shape[1]
    w = np.random.rand(n)
        
    #Arrays for storing values of cost and weights
    costs = np.empty(iters)
    weights = np.empty((iters, n))
    
    #Loop for Gradient Descent
    for i in range(iters):
    
        #H is m*1 vector of hypothesis function results
        H = 1 / (1 + np.exp(-np.dot(X,w))) 
        
        #Calculating new weights/costs using step down
        w = w - alpha * (X.T.dot(H - y)) / m
        j = costFunction(w, X, y)
        
        #Updating index for new weights/costs
        weights[i, :] = w.T
        costs[i] = j
        
    #Finding the best weights of all iterations
    best = np.where(costs==min(costs))
    w = weights[best]
    
    #Plot here
    plt.title("Gradient Descent")
    plt.xlabel("Iterations (i)")
    plt.ylabel("Cost (J)")
    plt.plot(costs)
    plt.scatter(best, costs[best], color="red")
    plt.show()
    
    return w.reshape(n,)


def truePositive(H, y):
    
    tp = 0
    
    for i in range(len(y)):
        if(H[i] == y[i] and y[i]==1):
            tp = tp + 1
            
    return tp

def trueNegative(H, y):
    
    tn = 0
    
    for i in range(len(y)):
        if(H[i] == y[i] and y[i]==0):
            tn = tn + 1
            
    return tn

def falsePositive(H, y):
    
    fp = 0
    
    for i in range(len(y)):
        if(H[i] != y[i] and H[i]==1):
            fp = fp + 1
            
    return fp

def falseNegative(H, y):
    
    fn = 0
    
    for i in range(len(y)):
        if(H[i] != y[i] and H[i]==0):
            fn = fn + 1
            
    return fn

def confusionMatrix(w, X, y):
    
    #H is m*1 vector of hypothesis function results
    H = 1 / (1 + np.exp(-np.dot(X,w))) 
    H = np.where(H >= 0.5, 1, 0)
    
    #Calculating Confusion Matrix
    TP = truePositive(H, y)
    TN = trueNegative(H, y)
    FP = falsePositive(H, y)
    FN = falseNegative(H, y)
    
    #Calculating Statistics
    accuracy = (TP + TN) / len(H)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("\t\tPredicted Yes\tPredicted No")
    print("Actual Yes\t TP = " + str(TP) + "\t FN = " + str(FN) + "\n")
    print("Actual No\t FP = " + str(FP) + "\t\t TN = " + str(TN) + "\n")
    
    print("Accuracy = " + str(accuracy))
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))
    print("F1 Score = " + str(f1))
    
    return

#Opening training file and reading raw data
fileName = input("Please enter the name of your training data file: ")
trFile = open(fileName, "r")
trData = np.loadtxt(fileName, skiprows=1, delimiter='\t', dtype='float')

# MxN for the training set
train_m = trData.shape[0]
train_n = trData.shape[1] - 1

#Getting X, y, and weights for training set
ones = np.ones((train_m, 1))
train_X = np.concatenate((ones, trData[:, 0:train_n]), axis = 1)
train_y = trData[:, train_n]
train_w = gradientDescent(train_X, train_y, alpha=0.01, iters=1000)

#Opening test file and reading raw data
fileName = input("Please enter the name of your test data file: ")
teFile = open(fileName, "r")
teData = np.loadtxt(fileName, skiprows=1, delimiter='\t', dtype='float')

# MxN for the test set
test_m = teData.shape[0]
test_n = teData.shape[1] - 1

#Getting X, y, and weights for training set
ones = np.ones((test_m, 1))
test_X = np.concatenate((ones, teData[:, 0:test_n]), axis = 1)
test_y = teData[:, test_n]
test_J = costFunction(train_w, test_X, test_y)

#Printing Final J and Confusion Matrix/Statistics
print("Weights = " + str(train_w))
print("Train J = " + str(costFunction(train_w, train_X, train_y)))
print("\nFinal J = " + str(test_J) + "\n")
confusionMatrix(train_w, test_X, test_y)