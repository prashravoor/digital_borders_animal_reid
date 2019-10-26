import numpy as np
import pymongo as pm
import cv2
import cv2.dnn as dnn
import matplotlib.pyplot as plt
import os
import sys
from db_interface import DbRecord,DbInterface
# import pickle
# from bson.binary import Binary

def initializeDB(client, dbName):
    cursor = client.getDB(dbName)
    collections = cursor.getCollectionNames()
    if len(collections) > 0:
        char = input('DB {} already contains some records. Reomve them? (y/n)'.format(dbName))
        if char.lower()[0] == 'y':
            for col in collections:
                cursor.dropCollection(col)

def readCaffeModel(modelName):
    modelCfg = 'models/{}.prototxt'.format(modelName)
    modelWeights = 'models/{}.caffemodel'.format(modelName)
    return dnn.readNetFromCaffe(modelCfg, modelWeights)

def readImageFromFile(filename):
    return cv2.imread(filename)

def createBlobFromImage(image, newSize, scale=(1.0/255), meanSubtract=(0,0,0), swapRB=True, crop=False):
    return dnn.blobFromImage(image, scale, newSize, meanSubtract, swapRB=swapRB, crop=crop)

def getNetOutput(net, inputBlob, layerName=None):
    net.setInput(inputBlob)
    return net.forward(layerName) 

def getNetOutputs(net, inputBlob, layerNames):
    net.setInput(inputBlob)
    return net.forward(layerNames)

def extractFeaturesForImage(net, modelName, layerNames, imageFile):
    inputSize = (227,227)
    if 'googlenet' == modelName:
        inputSize = (224,224)
    elif 'resnet50' == modelName:
        inputSize = (256,256)
    
    image = readImageFromFile(imageFile)
    blob = createBlobFromImage(image, inputSize)
    return getNetOutputs(net, blob, layerNames)

def processImagesAndStoreFeatures(client, datasetName, modelName, modelsDict, imagesList, idsList):
    network = modelsDict[modelName]
    layerNames = network.getLayerNames()
    cursor = client.getDB(getDbName(datasetName, modelName))

    for i in range(len(imagesList)):
        features = extractFeaturesForImage(network, modelName, layerNames, imagesList[i])
        for j in range(len(features)):
            rec = DbRecord(idsList[i], imagesList[i], features[j])
            cursor.insertAsync(layerNames[j], rec)

def readIdsForImages(imagesList, setType, keyfile):
    ids = []
    mapping = {}
    with open(keyfile) as f:
        idpairs = list(map(lambda x: x.strip().split('\t'), f.readlines()))
        f.close()
    
    for pair in idpairs:
        mapping[pair[0]] = int(pair[1])
        
    if setType == 'amur':
        for image in imagesList:
            base = os.path.basename(image)
            ids.append(mapping[(base.split('.')[0])])
    else:
        for image in imagesList:
            base = os.path.basename(image)
            ids.append(mapping[base.split('_')[0]])
    return ids


def getDbName(datasetName, modelName):
    return datasetName + '_' + modelName

def extractFeaturesForImages(args):
    if len(args) < 1:
        print('At least 1 images folder required')
        return
    client = DbInterface() 
    imageFolders = args

    # modelNames = ['alexnet', 'googlenet', 'resnet50']
    modelNames = ['alexnet', 'googlenet']
    modelsDict = {}
    for model in modelNames:
        modelsDict[model] = readCaffeModel(model)
        print('Loaded Model {}'.format(model))
    
    for i in range(len(imageFolders)):
        # Read Images from folder
        path = imageFolders[i]
        # Don't recurse into sub-folders, so get only reg files
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # Get only JPGs
        imageList = list(filter(lambda x: os.path.splitext(x)[1] == '.jpg', files))
        imageList = list(map(lambda x: os.path.join(path, x), imageList))
        if 'amur' in path.lower():
            imType = 'amur'
        else:
            imType = 'elp'
            
        keyfile = path + '/class_mapping.txt'
        if not os.path.exists(keyfile):
            print('No KeyFile (class_mapping.txt) for folder {}, skipping!'.format(path))
            continue

        imageIds = readIdsForImages(imageList, imType, keyfile)
        print('Found {} images in folder {}, Using Image Type {}'.format(len(imageList), path, imType))

        for modelName in modelNames:
            initializeDB(client, getDbName(imType, modelName))
            print('Processing {} images using Model {} for Set Type {}'.format(len(imageList), modelName, imType))
            processImagesAndStoreFeatures(client, imType, modelName, modelsDict, imageList, imageIds)

if __name__ == '__main__':
    extractFeaturesForImages(sys.argv[1:])

