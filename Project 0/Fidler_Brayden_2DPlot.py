# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 08:38:27 2020

@author: bfidler
"""

import numpy as np
import matplotlib.pyplot as plt

#Opening Iris file
irisFile = open("_IrisData.txt", "r")

#Creating arrays for each type of flower
seto = np.array([])
vers = np.array([])
virg = np.array([])

#Reading data file line by line
for line in irisFile.readlines():
    
    line = line.split()
    sepalLn = float(line[0])
    petalLn = float(line[2])
    flowerNm = line[4]
    
    #Assigning sepal and petal lengths to the correct array
    if flowerNm == 'setosa':
        seto = np.append(seto, (sepalLn, petalLn))
    elif flowerNm == 'versicolor':
        vers = np.append(vers, (sepalLn, petalLn))
    elif flowerNm == 'virginica':
        virg = np.append(virg, (sepalLn, petalLn))

#Reshaping arrays to have 2 columns
seto = seto.reshape(-1, 2)
vers = vers.reshape(-1, 2)
virg = virg.reshape(-1, 2)

#Column Identifiers
sepalLnCol = 0
petalLnCol = 1

#Creating axes to plot
plt.figure(figsize=(11,8.5))
irisPlt = plt.axes()
irisPlt.set(title="Iris Flowers",
            xlabel="Sepal Length",
            ylabel="Petal Length")

#Plotting flowers
irisPlt.scatter(seto[:, sepalLnCol], seto[:, petalLnCol], marker="*",
                color ="red", label="Setosa")
irisPlt.scatter(vers[:, sepalLnCol], vers[:, petalLnCol], marker="+",
                color="blue", label="Versicolor")
irisPlt.scatter(virg[:, sepalLnCol], virg[:, petalLnCol], marker="x",
                color="green", label="Virginica")
irisPlt.legend()

#Saving and Showing Plot
plt.savefig("Fidler_Brayden_MyPlot.png")
plt.show()
irisFile.close()




