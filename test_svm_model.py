from train_validate_svm import *
from extract_features_and_store_to_mongo import *
import numpy as np
from db_interface import DbInterface,DbRecord
from joblib import dump, load
from sklearn.decomposition import IncrementalPCA
import sys
import os
from object_detection import ObjectDetector
import time
import argparse

'''
if not len(args) == 3:
    print('Usage: {} <Model path> <Folder to test>'.format(args[0]))
    exit(1)

modelPath = args[1]
folder = args[2]
'''

def find_acc(modelPath, folder, numImages, numIds, minSamplesPerId, samplesPerId, det):
# Load SVM model
    svm_model,pca_model,trained_ids = load(modelPath)

    imageList = [x for x in os.listdir(folder + '/test') if x.endswith('.jpg')]

# Load Ids
    id_map = {}
    with open('{}/normalized_class_mapping.txt'.format(folder)) as f:
        id_map = {x.split('\t')[0].strip(): int(x.split('\t')[1].strip()) for x in f.readlines()}

    rev_map = {}
    for k,v in id_map.items():
        if not v in rev_map:
            rev_map[v] = []
        if k in imageList:
            rev_map[v].append(k)

    print('Created reverse map, containing {} ids'.format(len(rev_map)))
# Filter out all ids which have less than imagesPerId
    rev_map = [(k,v) for k,v in rev_map.items() if len(v) >= minSamplesPerId]
    rev_map = sorted(rev_map, key=lambda x: -len(x[1])) # Sort ids in descending order of num images
    print('After filtering out ids having less than {} images, total ids: {}'.format(minSamplesPerId, len(rev_map)))

    np.random.seed(22009)
# Remove random elements if more than samplesPerId images exist
    if samplesPerId is not None:
        if samplesPerId < minSamplesPerId:
            print('Invalid. Samples Per Id {} less than Min Samples Per Id: {}'.format(samplesPerId, minSamplesPerId))
            return None,None,None

        # Remove any ids which have fewer than samplesPerId
        rev_map = [x for x in rev_map if len(x[1]) >= samplesPerId]
        if numIds is not None and len(rev_map) < numIds:
            print('Not enough samples found. Only {} ids have more than {} images'.format(len(rev_map), samplesPerId))
            return None,None,None

        for i in range(len(rev_map)):
            rev_map[i] = (rev_map[i][0], np.random.choice(rev_map[i][1], samplesPerId))

        print('Found {} which have {} images per id'.format(len(rev_map), samplesPerId))

    if numIds is not None:
        if len(rev_map) < numIds:
            print('Not enough ids left after filtering.')
            return None,None,None

        pick = np.random.choice(len(rev_map), numIds, replace=False)
        rev_map = [rev_map[x] for x in pick]
        print('Retaining {} ids'.format(len(rev_map)))

    imageList = []
    for item in rev_map:
        imageList.extend([os.path.join(folder + '/test', x) for x in item[1]])
    np.random.shuffle(imageList)

    print('Proceeding with {} Identities, covering {} images overall'.format(len(rev_map), len(imageList)))

# Get Layer name and model name from SVM model
    parts = os.path.basename(modelPath).split('.')[0].split('-')
    dsName = parts[0]
    modelName = parts[1]
    layerName = parts[2].replace('&', '/')
    kernel = parts[3]
    transform = parts[4]
    if 'None' == transform:
        transform = None

# Load classification model and warm up
    classifier = readCaffeModel(modelName)
    extractFeaturesForImage(classifier, modelName, [layerName], imageList[0], det)
    extractFeaturesForImage(classifier, modelName, [layerName], imageList[0], det)

    print('Successfully loaded classifier {}'.format(modelName))
    print('Using {} layer for identification, Kernel: {}, Transform: {}'.format(layerName, kernel, transform))

    count = 0
    correct = 0
    times = 0
    class_times = 0
    for f in imageList:
        if not int(id_map[os.path.basename(f)]) in trained_ids:
            continue
        start = time.time()
        feature = extractFeaturesForImage(classifier, modelName, [layerName], f, det)[0]
        class_times += time.time() - start
        feature = normalize_activations(feature)
        flattened = feature.flatten().reshape([1,-1])
        if not pca_model is None:
            feature = pca_model.transform(flattened)

        animalId = svm_model.predict(feature)[0]
        times += time.time() - start
        correct_id = int(id_map[os.path.basename(f)])
        #print('Predicted Id: {}, Actual Id: {}'.format(animalId, correct_id))
        count += 1
        if correct_id == animalId:
            correct += 1

    return correct,count,times,class_times

if '__main__' == __name__:
    args = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgFolder', metavar='i', type=str,
                        help='images folder', required=True)
    parser.add_argument('--modelPath', metavar='m', type=str,
                        help='images folder', required=True)
    parser.add_argument('--numImages', metavar='n', type=int,
                        help='Max Images to test', default=1000)
    parser.add_argument('--numIds', metavar='id', type=int,
                        help='Number of Ids to use')
    parser.add_argument('--minSamplesPerId', metavar='ms', type=int,
                        help='Min Samples to use per id', default=5)
    parser.add_argument('--samplesPerId', metavar='s', type=int,
                        help='Samples to use per id')

    parsed = parser.parse_args(args[1:])
    modelPath = parsed.modelPath
    folder = parsed.imgFolder
    numImages = parsed.numImages
    numIds = parsed.numIds
    minSamplesPerId = parsed.minSamplesPerId
    samplesPerId = parsed.samplesPerId
    print(parsed)

    print('Loading object detector...')
    det = ObjectDetector('ssd/saved_model')
    det.loadModel()

    print('Warming up...')
# Warmup
    #det.getBoundingBoxes('amur_small/002019.jpg')
    #det.getBoundingBoxes('amur_small/002019.jpg')

    correct,count,times = find_acc(modelPath,folder,numImages,numIds,minSamplesPerId,samplesPerId, det)

    print('---- Results -----')
    #if count > 0:
    #    print('Correct: {}, Accuracy over {} images: {:.3f}'.format(correct, count, correct/count))
    #    print('Average Identification time taken per image: {:.3f}s'.format(times/count))

    '''
    with open('{}_{}_{}_{}_stats.csv'.format(dsName, modelName, numIds, samplesPerId), 'w') as f:
        f.write('{},{},{},{:.3f}'.format(numIds, count, correct, correct/count))
    '''
