from extract_features_and_store_to_mongo import *
import numpy as np
from db_interface import DbInterface,DbRecord
from joblib import dump, load
from sklearn.decomposition import IncrementalPCA
import sys
import os
from object_detection import ObjectDetector
import time

args = sys.argv
if not len(args) == 3:
    print('Usage: {} <Model path> <Folder to test>'.format(args[0]))
    exit(1)

modelPath = args[1]
folder = args[2]

# Load SVM model
svm_model,pca_model,trained_ids = load(modelPath)

imageList = [os.path.join(folder,x) for x in os.listdir(folder) if x.endswith('.jpg')]
n = min(30, len(imageList))
imageList = imageList[:n]

# Load Ids
id_map = {}
with open('{}/normalized_class_mapping.txt'.format(folder)) as f:
    id_map = {x.split('\t')[0].strip(): x.split('\t')[1].strip() for x in f.readlines()}

# Get Layer name and model name from SVM model
parts = os.path.basename(modelPath).split('.')[0].split('-')
modelName = parts[1]
layerName = parts[2].replace('&', '/')
kernel = parts[3]
transform = parts[4]
if 'None' == transform:
    transform = None

# Load classification model
classifier = readCaffeModel(modelName)
print('Successfully loaded classifier {}'.format(modelName))
print('Using {} layer for identification, Kernel: {}, Transform: {}'.format(layerName, kernel, transform))

print('Loading object detector...')
det = ObjectDetector('ssd/saved_model')
det.loadModel()

print('Warming up...')
# Warmup
det.getBoundingBoxes(readImageFromFile(imageList[0]))
det.getBoundingBoxes(readImageFromFile(imageList[0]))

print('Running test on {} images'.format(n))
count = 0
correct = 0
times = 0
for f in imageList:
    if not int(id_map[os.path.basename(f)]) in trained_ids:
        continue
    feature = extractFeaturesForImage(classifier, modelName, [layerName], f, det)[0]
    start = time.time()
    feature = normalize_activations(feature)
    flattened = feature.flatten().reshape([1,-1])
    if not pca_model is None:
        feature = pca_model.transform(flattened)

    animalId = svm_model.predict(feature)[0]
    times += time.time() - start
    correct_id = id_map[os.path.basename(f)]
    print('Predicted Id: {}, Actual Id: {}'.format(animalId, correct_id))
    count += 1
    if correct_id == animalId:
        correct += 1

print('---- Results -----')
if count > 0:
    print('Accuracy over {} images: {:.3f}'.format(n, correct/count))
    print('Average Identification time taken per image: {:.3f}s'.format(times/count))
