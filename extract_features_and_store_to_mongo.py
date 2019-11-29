import numpy as np
import pymongo as pm
import cv2
import cv2.dnn as dnn
import matplotlib.pyplot as plt
import os
import sys
from db_interface import DbRecord,DbInterface
from object_detection import ObjectDetector
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def initializeDB(client, dbName):
    cursor = client.getDB(dbName)
    collections = cursor.getCollectionNames()
    if len(collections) > 0:
        # char = input('DB {} already contains some records. Reomve them? (y/n)'.format(dbName))
        char = 'y'
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

def extractFeaturesForImage(net, modelName, layerNames, imageFile, detector):
    inputSize = (227,227)
    if 'googlenet' == modelName:
        inputSize = (224,224)
    elif 'resnet50' == modelName:
        inputSize = (224,224)
    
    image = readImageFromFile(imageFile)
    if image is None:
        print('Invalid imageFile: {}'.format(imageFile))
        exit(1)

    bounding_boxes = detector.getBoundingBoxes(image)
    if len(bounding_boxes) == 1:
        box = bounding_boxes[0].bounding_box
        image = image[box.ymin:box.ymax, box.xmin:box.xmax,:]

    blob = createBlobFromImage(image, inputSize)
    return getNetOutputs(net, blob, layerNames)

def inAny(val, values):
    val = val.lower()
    for v in values:
        if v.lower() in val:
            return True
    return False

def perform_global_average_pooling(feature):
    #if len(feature.shape) >= 3:
    #    return np.mean(feature, axis=1)
    return feature

def normalize_activations(feature):
    orig_shape = feature.shape
    feature = feature.flatten().reshape((-1,1)) # Convert to single sample of 1d array
    #feature = MinMaxScaler(feature_range=(-1,1)).fit_transform(feature)
    feature = StandardScaler().fit_transform(feature)
    return feature.reshape(orig_shape)

def processImagesAndStoreFeatures(client, datasetName, modelName, modelsDict, imagesList, idsList, detector):
    network = modelsDict[modelName]
    layerNames = network.getLayerNames()
    cursor = client.getDB(getDbName(datasetName, modelName))

    pick_layers = ['pool', 'relu', 'fc', 'output']
    subset = False
    if len(layerNames) > 100:
        print('Reducing number of layers. Only {} lasyers out of {} will be used'.format(pick_layers, len(network.getLayerNames())))
        subset = True

    for i in range(len(imagesList)):
        features = extractFeaturesForImage(network, modelName, layerNames, imagesList[i], detector)
        if subset:
            features = [features[x] for x in range(len(features)) if inAny(layerNames[x], pick_layers)]

        # Normalize activations
        features = [normalize_activations(x) for x in features]
        for y in range(len(features)):
            f = features[y]
            #assert ((f >= 0).all() and (f <= 1).all())
            assert not (f == float('nan')).any()

        for j in range(len(features)):
            rec = DbRecord(idsList[i], imagesList[i], perform_global_average_pooling(features[j]))
            cursor.insertAsync(layerNames[j], rec)

def readIdsForImages(imagesList, setType, keyfile):
    ids = []
    mapping = {}
    with open(keyfile) as f:
        idpairs = list(map(lambda x: x.strip().split('\t'), f.readlines()))
        f.close()
    
    for pair in idpairs:
        mapping[pair[0]] = int(pair[1])

    for image in imagesList:
        base = os.path.basename(image)
        ids.append(mapping[base])

    return ids

def getDbName(datasetName, modelName):
    return datasetName + '_' + modelName

def randomized_samples(path, keyfile='normalized_class_mapping.txt', max_files=600, seed=2020, min_images=3):
    with open('{}/{}'.format(path, keyfile)) as f:
        id_map = [tuple(x.strip().split('\t')) for x in f.readlines()]

    id_map = [x for x in id_map if os.path.exists(os.path.join(path,x[0]))]

    if len(id_map) < max_files:
        return [x[0] for x in id_map]

    reverse_map = {}
    for i in id_map:
        if not i[1] in reverse_map:
            reverse_map[i[1]] = []
        reverse_map[i[1]].append(i[0])

    id_lens = [[k, v] for k,v in reverse_map.items() if len(v) >= min_images]
    id_lens = sorted(id_lens, key=lambda x: -len(x[1])) # Sort in descending order of lengths
    i = 0
    np.random.seed(seed)

    while sum([len(x[1]) for x in id_lens]) > max_files:
        if i > len(id_lens):
            i = 0

        if i == 0 and len(id_lens[i][1]) < 4:
            # Remove some ids
            cur_files = sum([len(x[1]) for x in id_lens])
            n = int((cur_files - max_files)/3) # At this point, each id has max 3 images
            pick = np.random.choice(range(len(id_lens)), n)
            id_lens = [id_lens[x] for x in range(len(id_lens)) if x not in pick]
            break

        if len(id_lens[i]) == 2 and len(id_lens[i][1]) < 4:
            i = 0 # Reset and start from beginning
            continue

        n = max(1, int(.25 * len(id_lens[i][1])))
        pick = np.random.choice(id_lens[i][1], n)
        # Modify entry in place, remove all elements from id_lens which are in pick
        id_lens[i][1] = [x for x in id_lens[i][1] if x not in pick]
        i = (i + 1) % len(id_lens)

    imageList = []
    for x in id_lens:
        imageList.extend(x[1])

    return [os.path.join(path,x) for x in imageList]


def run_job(path, det, modelsDict, modelNames):
    # Read Images from folder
    # Don't recurse into sub-folders, so get only reg files
    # files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #files = files[:min(n,len(files))]
    # Get only JPGs
    #imageList = list(filter(lambda x: os.path.splitext(x)[1] == '.jpg', files))
    #imageList = list(map(lambda x: os.path.join(path, x), imageList))
    # Get random subset not exceeding 600 images
    imageList = randomized_samples(path)
    print('Reading in features for {} images'.format(len(imageList)))

    if 'amur' in path.lower():
        imType = 'amur'
    elif 'jaguar' in path.lower():
        imType = 'jaguar'
    else:
        imType = 'elp'
        
    keyfile = path + '/normalized_class_mapping.txt'
    if not os.path.exists(keyfile):
        print('No KeyFile (class_mapping.txt) for folder {}, skipping!'.format(path))
        return

    imageIds = readIdsForImages(imageList, imType, keyfile)
    print('Found {} images in folder {}, Using Image Type {}'.format(len(imageList), path, imType))

    client = DbInterface() 

    for modelName in modelNames:
        initializeDB(client, getDbName(imType, modelName))
        print('Processing {} images using Model {} for Set Type {}'.format(len(imageList), modelName, imType))
        processImagesAndStoreFeatures(client, imType, modelName, modelsDict, imageList, imageIds, det)


def extractFeaturesForImages(modelNames, args):
    if len(args) < 1:
        print('At least 1 images folder required')
        return

    imageFolders = args
    n = 5000
    if len(args) >= 2:
        try:
            n = int(args[-1])
            imageFolders = imageFolders[:len(imageFolders)-1]
        except:
            pass

    modelsDict = {}
    for model in modelNames:
        modelsDict[model] = readCaffeModel(model)
        print('Loaded Model {}'.format(model))

    print('Loading object Detection model...')
    det = ObjectDetector('ssd/saved_model')
    det.loadModel()
    print('Warming up object detector...')
    det.getBoundingBoxes(cv2.imread('amur_small/002019.jpg'))
    det.getBoundingBoxes(cv2.imread('amur_small/002019.jpg'))

    for i in range(len(imageFolders)):
        print('Starting Feature Extraction for folder {}'.format(imageFolders[i]))
        run_job(imageFolders[i], det, modelsDict, modelNames)

if __name__ == '__main__':
    # modelNames = ['alexnet', 'googlenet', 'resnet50']
    modelNames = ['alexnet']
    extractFeaturesForImages(modelNames, sys.argv[1:])

