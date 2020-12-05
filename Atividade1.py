# -*- coding: utf-8 -*-

import cv2
from skimage import feature
from sklearn.svm import SVC
import numpy as np
import pickle
from tqdm import tqdm
from os import listdir
from os.path import isfile, join


rootFolder = "atv1\\"
trainFolder = "treino\\"
testFolder = "teste\\"
obj1Folder = "carro\\"
obj2Folder = "moto\\"
obj3Folder = "onibus\\"

cropTrainFolder = "train\\"
cropTestFolder = "test\\"

svmFilename = "obj.svm"

widthWindow = 400
heightWindow = 400

orientationsParam = 9
pixelsPerCellParam = 4
cellsPerBlockParam = 2

def extractFilenamesFromFolder(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

def cropRegionOfEachPosImage(path, trainFolder, cropFolder, listOfImages, isTrain):
    i = 0
    if(isTrain == 1):
        print("Montando base de imagens de carro para treino.")
    elif(isTrain == 2):
        print("Montando base de imagens de moto para teste.")
    elif(isTrain == 3):
        print("Montando base de imagens de onibus para teste.")

    for eachImage in tqdm(listOfImages):
        if(isTrain == 1):
            img = cv2.imread(path + trainFolder + obj1Folder + eachImage)
        elif(isTrain == 2):
            img = cv2.imread(path + trainFolder + obj2Folder + eachImage)
        elif(isTrain == 3):
            img = cv2.imread(path + trainFolder + obj3Folder + eachImage)
      
        img = cv2.resize(img, (widthWindow, heightWindow))
        flipImg = cv2.flip(img, 1)
        
        stringImg = listOfImages[i].split("/", 1)
        
        if(isTrain == 1):
            cv2.imwrite(path + cropFolder + obj1Folder + stringImg[0], img)
            cv2.imwrite(path + cropFolder + obj1Folder + "flip_" + str(i) + ".png", flipImg)
        elif(isTrain == 2):
            cv2.imwrite(path + cropFolder + obj2Folder + stringImg[0], img)
            cv2.imwrite(path + cropFolder + obj2Folder + "flip_" + str(i) + ".png", flipImg)
        elif(isTrain == 3):
            cv2.imwrite(path + cropFolder + obj3Folder + stringImg[0], img)
            cv2.imwrite(path + cropFolder + obj3Folder + "flip_" + str(i) + ".png", flipImg)
       

        i += 1

def setDatabase():
    #train carro
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + trainFolder + obj1Folder)
    cropRegionOfEachPosImage(rootFolder, trainFolder, cropTrainFolder, vectorOfFilenameImagesPos, 1)

    #train moto
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + trainFolder + obj2Folder)
    cropRegionOfEachPosImage(rootFolder, trainFolder, cropTrainFolder, vectorOfFilenameImagesPos, 2)

    #train onibus
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + trainFolder + obj3Folder)
    cropRegionOfEachPosImage(rootFolder, trainFolder, cropTrainFolder, vectorOfFilenameImagesPos, 3)




def trainSVM():
    X = np.empty(0)
    Y = np.empty(0)
    dimensionOfFeatureVector = 0
    
    print("Extraindo feature HOG de carro.")
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + cropTrainFolder + obj1Folder)
    for eachImage in tqdm(vectorOfFilenameImagesPos):
        eachImage = eachImage.split("/", -1)
        eachImage = rootFolder + cropTrainFolder + obj1Folder + eachImage[0]
        
        img = cv2.imread(eachImage, 0)
        img = cv2.resize(img, (widthWindow, heightWindow))
        H = feature.hog(img, orientations=orientationsParam, pixels_per_cell=(pixelsPerCellParam, pixelsPerCellParam), cells_per_block=(cellsPerBlockParam, cellsPerBlockParam), block_norm='L2-Hys', feature_vector = True)
        H = np.array(H)
        
        if(X.shape[0] == 0):
            X = H
            dimensionOfFeatureVector = X.shape[0]
        else:
            X = np.append(X, H)
            
        if(Y.shape[0] == 0):
            Y = np.array([0])
        else:
            Y = np.append(Y, np.array([0]))
             
    print("Extraindo feature HOG de moto.")
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + cropTrainFolder + obj2Folder)
    for eachImage in tqdm(vectorOfFilenameImagesPos):
        eachImage = eachImage.split("/", -1)
        eachImage = rootFolder + cropTrainFolder + obj2Folder + eachImage[0]
        
        img = cv2.imread(eachImage, 0)
        img = cv2.resize(img, (widthWindow, heightWindow))
        H = feature.hog(img, orientations=orientationsParam, pixels_per_cell=(pixelsPerCellParam, pixelsPerCellParam), cells_per_block=(cellsPerBlockParam, cellsPerBlockParam), block_norm='L2-Hys', feature_vector = True)
        H = np.array(H)
        
        X = np.append(X, H)            
        Y = np.append(Y, np.array([1]))
             
    print("Extraindo feature HOG de onibus.")
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + cropTrainFolder + obj3Folder)
    for eachImage in tqdm(vectorOfFilenameImagesPos):
        eachImage = eachImage.split("/", -1)
        eachImage = rootFolder + cropTrainFolder + obj3Folder + eachImage[0]
        
        img = cv2.imread(eachImage, 0)
        img = cv2.resize(img, (widthWindow, heightWindow))
        H = feature.hog(img, orientations=orientationsParam, pixels_per_cell=(pixelsPerCellParam, pixelsPerCellParam), cells_per_block=(cellsPerBlockParam, cellsPerBlockParam), block_norm='L2-Hys', feature_vector = True)
        H = np.array(H)
        
        X = np.append(X, H)            
        Y = np.append(Y, np.array([2]))
        
    print("Treinando o SVM.\n")
    X = np.reshape(X, (-1, dimensionOfFeatureVector))

    svm = SVC(kernel='linear')
    svm.fit(X, Y)
    return svm

def testImages(svmObj):
    font = cv2.FONT_HERSHEY_SIMPLEX
    vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + testFolder)
    print("Classificando imagens.")
    for eachImage in tqdm(vectorOfFilenameImagesPos):
        eachImage = eachImage.split("/", -1)
        eachImage = rootFolder + testFolder + eachImage[0]
        
        img = cv2.imread(eachImage, 0)
        imgShow = cv2.imread(eachImage, 1)
        h, w, _ = imgShow.shape
        img = cv2.resize(img, (widthWindow, heightWindow))

        H = feature.hog(img, orientations=orientationsParam, pixels_per_cell=(pixelsPerCellParam, pixelsPerCellParam), cells_per_block=(cellsPerBlockParam, cellsPerBlockParam), block_norm='L2-Hys', feature_vector = True)
        X = np.array(H)
        X = np.reshape(X, (-1, X.shape[0]))
        
        if(svmObj.predict(X) == 0):
            cv2.putText(imgShow, 'Carro', (0, int(h * 0.15)), font, 1, (0, 255, 0), 3, cv2.LINE_AA)
        elif(svmObj.predict(X) == 1):
            cv2.putText(imgShow, 'Moto', (0, int(h * 0.15)), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        elif(svmObj.predict(X) == 2):
            cv2.putText(imgShow, 'Onibus', (0, int(h * 0.15)), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
        
        cv2.imshow("Resultado", imgShow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def saveSVM(obj):
    pickle.dump(obj, open(svmFilename, "wb"))    

def loadSVM():
    return pickle.load(open(svmFilename, "rb"))

def main():
    
    #constroi a base de dados
    #setDatabase()
    
    #treina o SVM
    #svmObj = trainSVM()
    #saveSVM(svmObj)
    
    #testa o SVM
    svmObj = loadSVM()
    testImages(svmObj)

if __name__ == "__main__":
    main()