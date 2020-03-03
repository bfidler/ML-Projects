# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:16:04 2020

@author: bfidler
"""
def quadratic(d):
    rows = d.shape[0]
    newD = np.empty((rows, ))
    
    for col in d.T[:-1]:
        sq = np.power(col, 2)
        newD = np.column_stack((newD, col))
        newD = np.column_stack((newD, sq))        
    
    newD = np.column_stack((newD, d.T[-1]))
    newD = np.delete(newD, 0, 1)
    return newD

def cubic(d):
    rows = d.shape[0]
    newD = np.empty((rows, ))
    
    for col in d.T[:-1]:
        sq = np.power(col, 2)
        cb = np.power(col, 3)
        newD = np.column_stack((newD, col))
        newD = np.column_stack((newD, sq)) 
        newD = np.column_stack((newD, cb))
    
    newD = np.column_stack((newD, d.T[-1]))
    newD = np.delete(newD, 0, 1)    
    return newD

def quartic(d):
    rows = d.shape[0]
    newD = np.empty((rows, ))
    
    for col in d.T[:-1]:
        sq = np.power(col, 2)
        cb = np.power(col, 3)
        qt = np.power(col, 4)
        newD = np.column_stack((newD, col))
        newD = np.column_stack((newD, sq)) 
        newD = np.column_stack((newD, cb))
        newD = np.column_stack((newD, qt))
    
    newD = np.column_stack((newD, d.T[-1]))
    newD = np.delete(newD, 0, 1)    
    return newD

def quintic(d):
    rows = d.shape[0]
    newD = np.empty((rows, ))
    
    for col in d.T[:-1]:
        sq = np.power(col, 2)
        cb = np.power(col, 3)
        qt = np.power(col, 4)
        qn = np.power(col, 5)
        newD = np.column_stack((newD, col))
        newD = np.column_stack((newD, sq)) 
        newD = np.column_stack((newD, cb))
        newD = np.column_stack((newD, qt))
        newD = np.column_stack((newD, qn))
    
    newD = np.column_stack((newD, d.T[-1]))
    newD = np.delete(newD, 0, 1)    
    return newD

import numpy as np


#Gather Data
fileName = input("Please enter the name of your data file: ")
file = open(fileName, "r")
data = np.loadtxt(fileName, skiprows=1, delimiter='\t', dtype='float')
file.close()

#Shuffle Data
np.random.shuffle(data)

#bikeData = quadratic(bikeData)
##bikeData = cubic(bikeData)
#bikeData = quartic(bikeData)
#bikeData = quintic(bikeData)

#Calculate # of rows for training and test
m = data.shape[0]
mTrain = int(m*.7) 
mTest =  m - mTrain

#Write 50% Training data
hdr = str(mTrain) + '\t' + str(data.shape[1] - 1)
np.savetxt("train.txt", data[0:mTrain], header=hdr, \
           fmt='%.7s', delimiter='\t', comments='')

#Write 50% Test data
hdr = str(mTest) + '\t' + str(data.shape[1] - 1)
np.savetxt("test.txt", data[mTrain:m], header=hdr, \
           fmt='%.7s', delimiter='\t', comments='')