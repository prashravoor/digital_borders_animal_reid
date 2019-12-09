from object_detection import ObjectDetector,BoundingBox
from extract_features_and_store_to_mongo import *
from train_validate_svm import *
import time
import os
from joblib import load
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import namedtuple
import cv2

DetectionResult = namedtuple('DetectionResult', 'bounding_box classid identifier')

class SvmIdentifier:
    def __init__(self, savedModelPath, objDet):
        parts = savedModelPath.split('-')
        if not len(parts) == 5:
            print('Invalid model {} provided'.format(savedModelPath))
            raise ValueError
        self.modelPath = savedModelPath
        self.dsName = os.path.basename(parts[0])

        if self.dsName == 'amur':
            self.expected_ids = [388, 446, 242, 413] #388 is the only correct one 
        elif self.dsName == 'elp':
            self.expected_ids = [448]
        else:
            self.expected_ids = [275, 451, 458]
 
        self.modelName = parts[1]
        if self.modelName == 'alexnet':
            self.size = (227,227)
        else:
            self.size = (224,224)

        self.kernelType = parts[3]
        self.layer = parts[2].replace('&', '/')
        self.transform = parts[4]
        if 'None' == self.transform:
            self.transform = None
        self.objDetector = objDet
        self.svmModel = None
        self.pcaModel = None
        self.trained_ids = []
        self.classifier = None

    def loadModel(self):
       self.svmModel, self.pcaModel, self.trained_ids = load(self.modelPath)
       self.classifier = readCaffeModel(self.modelName)
       print('Loaded SVM, PCA models, and classifer {}'.format(self.modelName))
    
    def predictIdsFromImage(self, image):
        if image is None:
            return None

        results = self.objDetector.getBoundingBoxes(image)
        detections = []
        for result in results:
            bbox = result.bounding_box
            det_id = None
            if result.classid in self.expected_ids:
                det_id = self.getIdFromBoundingBox(image, bbox)

            detections.append(DetectionResult(bbox, result.classid, det_id))

        return detections

    def getIdFromBoundingBox(self, image, bbox):
        det_id = None
        bbox_image = image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax,:]
        blob = createBlobFromImage(bbox_image, self.size)
        feature = getNetOutput(self.classifier, blob, self.layer)
        det_id = self.predictIdForFeature(feature)

        return det_id


    def predictId(self, imageFile):
        # feature = extractFeaturesForImage(self.classifier, self.modelName, [self.layer], imageFile, self.objDetector)[0]
        return self.predictIdsFromImage(cv2.imread(imageFile))

    def predictIdForFeature(self, feature):
        feature = normalize_activations(feature)
        flattened = feature.flatten().reshape([1,-1])
        if not self.pcaModel is None:
            flattened = self.pcaModel.transform(flattened)

        animalId = self.svmModel.predict(flattened)
        return animalId[0]

    def isPresent(self, testId):
        return testId in self.trained_ids

    def belongsToClass(self, classid):
        return classid in self.expected_ids

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('Usage: {} <Model Path> <image folder> [id to use]'.format(args[0]))
        exit(1)

    modelPath = args[1]
    folder = args[2]

    existingImages = [x for x in os.listdir(folder) if x.endswith('.jpg')]
    with open('{}/normalized_class_mapping.txt'.format(folder)) as f:
        id_map = [(x.split('\t')[0].strip(),int(x.split('\t')[1].strip())) for x in f.readlines()]
        id_map = [(x[0],x[1]) for x in id_map if x[0] in existingImages]
    minId,maxId = min([x[1] for x in id_map]), max([x[1] for x in id_map])

    idNum = None
    if len(args) >= 4:
        idNum = int(args[3])

    rev_map = {}
    for x in id_map:
        if not x[1] in rev_map:
            rev_map[x[1]] = []
        rev_map[x[1]].append(x[0])


    print('Loading Object Detector')
    det = ObjectDetector('ssd/saved_model')
    det.loadModel()
    # Warming up
    det.getBoundingBoxes(readImageFromFile(os.path.join(folder, rev_map[0][0])))
    det.getBoundingBoxes(readImageFromFile(os.path.join(folder, rev_map[0][0])))


    print('Loading Object Identifier')
    idn = SvmIdentifier(modelPath, det)
    idn.loadModel()

    if not idNum is None and not idn.isPresent(idNum):
        print('Provided Id {} is not present in trained model, will choose random Id')
        idNum = None

    if idNum is None:
        idNum = np.random.choice(list(idn.trained_ids))
 
    if not idNum in rev_map:
        print('Id {} not found in mapping'.format(idNum))
        exit(2)

    print('Found {} images for id {}'.format(len(rev_map[idNum]), idNum))

    count = 0
    correct = 0
    times = 0
    fig = plt.figure(figsize=(20,20), num='Id: {}'.format(idNum))
    gs = gridspec.GridSpec(3,3)

    for im in rev_map[idNum]:
        fullPath = os.path.join(folder, im)

        start = time.time()
        predictedId = idn.predictId(fullPath)
        times += (time.time() - start)

        count += 1
        if predictedId == idNum:
            correct += 1

        if count < 9:
            ax = plt.subplot(gs[count])
            ax.imshow(readImageFromFile(fullPath)[:,:,(2,1,0)])
            ax.set_title('Predicted Id: {}'.format(predictedId))
            plt.axis('off')

    print('Accuracy over {} images: {:.3f}'.format(count, correct/count))
    print('Average time taken for identification pipeline: {:.3f}'.format(times/count))

    plt.show()
