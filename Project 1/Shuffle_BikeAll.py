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


#Gather Bike Data
bikeFile = open("BikeAll.txt", "r")
bikeData = np.loadtxt(bikeFile, skiprows=1, delimiter='\t', dtype='float')
bikeFile.close()

#Shuffle Data
np.random.shuffle(bikeData)

#bikeData = quadratic(bikeData)
bikeData = cubic(bikeData)
#bikeData = quartic(bikeData)
#bikeData = quintic(bikeData)

#Write 60% Training data
hdr = '439\t' + str(bikeData.shape[1] - 1)
np.savetxt("Fidler_Brayden_Train.txt", bikeData[0:439], header=hdr, \
           fmt='%.7s', delimiter='\t', comments='')

#Write 20% Valid data
hdr = '146\t' + str(bikeData.shape[1] - 1)
np.savetxt("Fidler_Brayden_Valid.txt", bikeData[439:585], header=hdr, \
           fmt='%.7s', delimiter='\t', comments='')

#Write 20% Test data
hdr = '146\t' + str(bikeData.shape[1] - 1)
np.savetxt("Fidler_Brayden_Test.txt", bikeData[585:731], header=hdr, \
           fmt='%.7s', delimiter='\t', comments='')