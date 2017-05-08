#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:09:49 2017

@author: sezan92
"""



from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG
import matplotlib.pyplot as plt
#Importing Models

from sklearn.svm import SVC,NuSVC

#Data Preparation

trainFolder = '/home/sezan92/BanglaDigit/' 

Zero = trainFolder+'0'
One= trainFolder+'1'
Two = trainFolder+'2'
Three = trainFolder+'3'
Four = trainFolder+'4'
Five = trainFolder+'5'
Six = trainFolder+'6'
Seven = trainFolder+'7'
Eight = trainFolder+'8'
Nine = trainFolder+'9'
testFolder = trainFolder + 'Test'
trainData = []
responseData = []
testData = []
NumberList = []
ZeroImages = [ f for f in listdir(Zero) if isfile(join(Zero,f)) ]
OneImages = [ f for f in listdir(One) if isfile(join(One,f)) ]
TwoImages = [ f for f in listdir(Two) if isfile(join(Two,f)) ]
ThreeImages = [ f for f in listdir(Three) if isfile(join(Three,f)) ]
FourImages = [ f for f in listdir(Four) if isfile(join(Four,f)) ]
FiveImages = [ f for f in listdir(Five) if isfile(join(Five,f)) ]
SixImages = [ f for f in listdir(Six) if isfile(join(Six,f)) ]
SevenImages = [ f for f in listdir(Seven) if isfile(join(Seven,f)) ]
EightImages = [ f for f in listdir(Eight) if isfile(join(Eight,f)) ]
NineImages = [ f for f in listdir(Nine) if isfile(join(Nine,f)) ]
TestImages = [ f for f in listdir(testFolder) if isfile(join(testFolder,f)) ]


def ReadImages(ListName,FolderName,Label):
    global NumberList
    global responseData
    global trainData
    global hog
    global cv2
    global imutils
    global winSize
    global testData
    for image in ListName:
        img = cv2.imread(join(FolderName,image))
        img = cv2.resize(img,(28,28))   
        feature = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        trainData.append(feature.T)
        responseData.append(Label)
    



ReadImages(ZeroImages,Zero,0)
ReadImages(OneImages,One,1)
ReadImages(TwoImages,Two,2)
ReadImages(ThreeImages,Three,3)
ReadImages(FourImages,Four,4)
ReadImages(FiveImages,Five,5)
ReadImages(SixImages,Six,6)
ReadImages(SevenImages,Seven,7)
ReadImages(EightImages,Eight,8)
ReadImages(NineImages,Nine,9)

trainNp =np.float32(trainData)
responseNp =np.float32(responseData)
X = np.float32(trainNp)
y= np.float32(responseNp)

svm = NuSVC(kernel = 'linear',nu=0.2)
svm.fit(X,y)

for i in range(len(TestImages)):
    img = cv2.imread(join(testFolder,TestImages[i]))
    img2 = cv2.resize(img,(28,28))
    featuretest= HOG(cv2.cvtColor(img2,
                                  cv2.COLOR_RGB2GRAY))
    ytest = svm.predict(featuretest)
    plt.figure()
    plt.imshow(img)
    plt.title(str(ytest))
