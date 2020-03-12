# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:05:29 2020

@author: bfidler
"""

import numpy as np


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

#Removes punctiation/converts to lowercase
def cleanText(txt):
    
    txt = txt.lower() #all lower
    txt = txt.strip() #strip whitespace
    
    for ltr in txt:
        if ltr in """[]!.,"-—@;':#$%^&*()<>{}=|\+/_""": #Special Chars
            txt = txt.replace(ltr, " ")
    
    #Note - I inentionally let ! and ? remain to affect accuracy
    return txt

#Counts appearance of words in spam or ham emails
def countWords(words, spam, counted):
    
    for w in words:
        if w in counted: #Checking if word has been counted yet
            if spam == 1:
                counted[w][1] = counted[w][1] + 1 #Add to spam count for word
            else:
                counted[w][0] = counted[w][0] + 1 # Add to ham count for word
        else:
            if spam == 1:
                counted[w] = [0,1] #Start the spam count for word
            else:
                counted[w] = [1,0] #Start the ham count for word
                
    return counted #Return updated dictionary
    

#Creates a vocab of percentages for each of the counted words
def calcPercentages (k, counted, spam, ham):
    
    #Looping through words
    for key in counted:
        counted[key][0] = (counted[key][0] + k) / (2*k+ham) #ham
        counted[key][1] = (counted[key][1] + k) / (2*k+spam) #spam

    return counted

#Calculates the probability that a subject line is spam using a vocab
def calcProbability (subject, vocab, spam, ham):
    
    pbSpam = spam / (spam+ham) #P(S)
    pbHam = ham / (spam+ham) #P(^S)
    pbSl_Spam = 1 #P(SL|S)
    pbSl_Ham = 1 #P(SL|^SL)
    
    for word in vocab: #Loop through vocab   
        if word in subject: #Check if word exists in subject line
            pbSl_Ham = vocab[word][0] * pbSl_Ham 
            pbSl_Spam = vocab[word][1] * pbSl_Spam
        else:
            pbSl_Ham = (1 - vocab[word][0]) * pbSl_Ham
            pbSl_Spam = (1 - vocab[word][1]) * pbSl_Spam
            
    #Return P(S|SL)
    return 1 / (1 + np.exp(np.log(pbSl_Ham*pbHam) - np.log(pbSl_Spam*pbSpam)))

def confusionMatrix(H, y):
    
    #Calculating Confusion Matrix
    H = np.where(H >= 0.5, 1, 0)
    TP = truePositive(H, y)
    TN = trueNegative(H, y)
    FP = falsePositive(H, y)
    FN = falseNegative(H, y)
    
    #Calculating Statistics
    accuracy = (TP + TN) / len(H)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("\t\tPredicted Spam\tPredicted Ham")
    print("Actual Spam\t TP = " + str(TP) + "\t FN = " + str(FN) + "\n")
    print("Actual Ham\t FP = " + str(FP) + "\t TN = " + str(TN) + "\n")
    
    print("Accuracy = " + str(accuracy))
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))
    print("F1 Score = " + str(f1))
    
    return

#File Name Prompts
trName = input("Please enter the name of the training file: ")
swName = input("Please enter the name of the stop-word file: ")
teName = input("Please enter the name of the test file: ")

#Creating set of stop words from stop word file
swFile = open(swName, "r", encoding="unicode-escape")
stopWds = set(swFile.read().split())

#Closing stop word file and opening training file
swFile.close()
trainFile = open(trName, "r", encoding="ascii", errors="ignore")

#Initial Training Spam/Ham counts and empty dictionary
trSpam = 0
trHam = 0
trCount = dict()

#Reading training file by line
for line in trainFile.readlines(): 
   
    isSpam = int(line[0]) #Spam/Ham Label
    
    if isSpam == 1: #Update spam/ham counts
        trSpam = trSpam + 1
    else:
        trHam = trHam + 1
        
    line = cleanText(line) #Clean text
    line = line.split() #Split string into array
    line = set(line)  #Create set of words
    line = line.difference(stopWds) #Removing StopWords
    
    trCount = countWords(line, isSpam, trCount) #Update count dictionary
     
#Close training and open test file
trainFile.close()
testFile = open(teName, "r", encoding="ascii", errors="ignore")

#Test Spam/Ham Counts, predictions, and actual labels
teSpam = 0
teHam = 0
H = np.empty([])
y = []
vocab = calcPercentages(1, trCount, trSpam, trHam) #Vocab from training set

#Reading subject lines in test file
for line in testFile.readlines():
    
    #Spam label
    isSpam = int(line[0])
    y.append(isSpam)  
 
    if isSpam == 1: #Update spam/ham counts
        teSpam = teSpam + 1
    else:
        teHam = teHam + 1
        
    line = cleanText(line[1:]) #Clean text
    line = line.split() #Split string into array
    line = set(line)  #Create set of words
    line = line.difference(stopWds) #Removing StopWords
    
    #Prediction made
    pred = calcProbability(line, vocab, trSpam, trHam)
    H = np.append(H, pred)

#Printing count of spam and ham with confusion matrix
print("\nNumber of Spam Emails in Test Set: " + str(teSpam))
print("Number of Ham Emails in Test Set: " + str(teHam) + "\n")
confusionMatrix(H, y)
testFile.close()
