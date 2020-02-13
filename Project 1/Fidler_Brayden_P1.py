# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:43:32 2020

@author: bfidler
"""

def normalEquation(x, y):
    return np.dot(np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))

def costFunction(w, x, y):
    return np.sum((np.dot(x,w)- y) * (np.dot(x,w) - y)) / (2*y.size)

def r_squared(cost, y):
    y_avg = np.sum(y) / y.size
    ysum = np.sum(np.dot((np.subtract(y, y_avg)),(np.subtract(y, y_avg)))) 
    denom = ysum / (2 * y.size)
    return cost / denom

def adj_rsquared(r2, m, n):
    return 1 - ((1-r2)*(m-1)) / (m-n-1)

import numpy as np

#Opening training file and reading raw data
fileName = input("Please enter the name of your training data file: ")
trainingFile = open(fileName, "r")
trainingData = np.loadtxt(fileName, skiprows=1, delimiter='\t', dtype='float')

#Reading first line of training File
firstLine = trainingFile.readline()
numRows = int(firstLine.split()[0])
numFeatures = int(firstLine.split()[1])

#Preparing training data for Normal Equation
ones = np.ones((numRows, 1))
arr = trainingData[:,0:numFeatures]
trainingData_x = np.concatenate((ones, arr), axis=1)
trainingData_y = trainingData[:,numFeatures]

weights = normalEquation(trainingData_x, trainingData_y)
cost = costFunction(weights, trainingData_x, trainingData_y)

print("\nFinal W (weights) for training set: ")
for i in range(0, weights.size):
    print("x" + str(i) + ": " + str(weights[i]))
print("Final J (cost) for training set: " + str(cost))

#Opening validation file and reading raw data
fileName = input("Please enter the name of your validation data file: ")
validationFile = open(fileName, "r")
validationData = np.loadtxt(fileName, skiprows=1, delimiter='\t', dtype='float')

#Reading first line of validation File
firstLine = validationFile.readline()
numRows = int(firstLine.split()[0])
numFeatures = int(firstLine.split()[1])

#Preparing validation data for cost function
ones = np.ones((numRows, 1))
arr = validationData[:,0:numFeatures]
validationData_x = np.concatenate((ones, arr), axis=1)
validationData_y = validationData[:,numFeatures]

cost = costFunction(weights, validationData_x, validationData_y)
print("\nFinal J (cost) for validation set: " + str(cost))

#Opening test file and reading raw data
fileName = input("Please enter the name of your test data file: ")
testFile = open(fileName, "r")
testData = np.loadtxt(fileName, skiprows=1, delimiter='\t', dtype='float')

#Reading first line of test File
firstLine = testFile.readline()
numRows = int(firstLine.split()[0])
numFeatures = int(firstLine.split()[1])

#Preparing test data for cost function
ones = np.ones((numRows, 1))
arr = testData[:,0:numFeatures]
testData_x = np.concatenate((ones, arr), axis=1)
testData_y = testData[:,numFeatures]

cost = costFunction(weights, testData_x, testData_y)
rsquare = r_squared(cost, testData_y)
adj_rsquare = adj_rsquared(rsquare, numRows, numFeatures)

print("\nFinal J (cost) for test set: " + str(cost))
print("\nFinal Adjusted R^2 for test set: " + str(adj_rsquare))

