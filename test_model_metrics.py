from object_detection import ObjectDetector
from yolo_object_detection import YoloObjectDetector
import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from im_utils import drawBoundingBoxWithLabel
import matplotlib.gridspec as gridspec

def evaluateModel(model, imageFiles, expected_classid):
    runTimes = []
    confidences = []
    invalidClassIds = set()
    wrongImages = []
    detResults = []

    batchSize = 10
    yolo = False
    if isinstance(model, YoloObjectDetector):
        batchSize = 1
        yolo = True

    n = min(10000, len(imageFiles))
    imageFiles = imageFiles[:n]
    print('Running Inference on {} images, batching {} images at a time'.format(len(imageFiles), batchSize))
    for i in range(0,len(imageFiles),batchSize):
        n = min(i + batchSize, len(imageFiles))
        images = [cv2.imread(imageFile) for imageFile in imageFiles[i:n]]
        start = time.time()
        if yolo:
            results = [model.getBoundingBoxes(images[0])]
        else:
            results = model.getBatchDetectionResults(images)
        runTimes.append(time.time() - start)

        for j in range(len(results)):
            imageFile = imageFiles[i+j]
            result = results[j]
            wrong = False
            if len(result) == 0:
                print('Incorrect number of detections for image {}: {}'.format(imageFile, len(result)))
                wrong = True

            confidence = 0.
            if len(result) > 0:
                result = result[0]
                if not result.classid == expected_classid:
                    print('Incorrect detection for imageFile {}, Expected Class Id: {}, Actual: {}'
                            .format(imageFile, expected_classid, result.classid))
                    wrong = True
                    confidence = 0.
                    invalidClassIds.add(result.classid)
                else:
                    confidence = result.confidence

            confidences.append(confidence)

            if wrong:
                wrongImages.append(imageFile)
                detResults.append(results[j])

    return runTimes, confidences, invalidClassIds, detResults, wrongImages

def summarizeResult(images, runTimes, confidences, invalidClassIds, wrongDet, wrongImages):
    print()
    print('---- Summary ----')
    print('Total Images evaluated: {}, Wrong Detections: {}'.format(len(images), len(wrongImages)))
    print('Accuracy: {:.2f}'.format(1. - len(wrongImages)/float(len(images))))
    print('Total Runtime: {:.2f}, Average Runtime: {:.2f}s, Max Time: {:.2f}s, Min Time: {:.2f}s'
                .format(np.sum(runTimes), np.sum(runTimes)/len(images), np.max(runTimes), np.min(runTimes)))
    print('Average Confidence Score: {:.2f}, Max: {:.2f}, Min: {:.2f}'
                .format(np.average(confidences), np.max(confidences), np.min(confidences)))
    print('Invalid Class Id List: {}'.format(invalidClassIds))
    print()

def showMisclassifications(wrongImages, wrongDets, labels, title):

    if len(wrongImages) > 9:
        print('Showing first 9 misclassifications')
        wrongImages = wrongImages[:9]
        wrongDets = wrongDets[:9]

    rows = min(int(len(wrongImages) ** .5) + 1, 3)

    fig = plt.figure(figsize=(10,10))
    gs2 = gridspec.GridSpec(rows, rows)
    for i in range(len(wrongImages)):
        image = cv2.imread(wrongImages[i])
        ax = plt.subplot(gs2[i])
        # plt.axis('off')
        classes = []
        for res in wrongDets[i]:
            image = drawBoundingBoxWithLabel(image, res, labels, thickness=5)
            classes.append(labels[res.classid])
        image = cv2.resize(image, (300,300))
        ax.imshow(image[:,:,(2,1,0)])
        ax.set_title(os.path.basename(wrongImages[i]))
        ax.set_xlabel('Detections: {}'.format(classes))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.set_window_title(title)
    fig.tight_layout()

def loadYoloLabels(file):
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]
        f.close()

    return lines

def loadSsdLabels(file):
    with open(file) as f:
        lines = [x.split(',') for x in f.readlines()]
        f.close()

    mapping = {}
    for line in lines:
        mapping[int(line[0])] = line[1].strip()

    return mapping

def runModelMetrics(model, images, labels, expected_classid, name):
    print('Evaluating {} images...'.format(len(images)))
    print('Using {} Object Detector...'.format(name))
    print()

    print('Loading Model...')
    model.loadModel()

    # Warmup
    print('Warming up model...')
    image = cv2.imread(images[0])
    model.getBoundingBoxes(image)
    model.getBoundingBoxes(image)

    res = evaluateModel(model, images, expected_classid)
    summarizeResult(images, *res)

    wrongImages = res[-1]
    wrongDets = res[-2]

    if len(wrongImages) == 0:
        print('No Misclassifications for {}'.format(title))
    else:
        print('Misclassifications for {}'.format(name))
        showMisclassifications(wrongImages, wrongDets, labels, '{} Misclassifications'.format(name))


args = sys.argv
if len(args) < 2 or len(args) > 3:
    print('Usage: {} <image folder> [Max Images]'.format(args[0]))
    exit(1)

folder = args[1]
images = ['{}/{}'.format(folder, x) for x in os.listdir(folder) if x.endswith('.jpg')]

if len(args) == 3:
    maxImages = int(args[2])
    images = images[:maxImages]

yolo_labels = loadYoloLabels('../darknet/darknet/data/coco.names')
yolo_expected_classid = 20 # Elephant. Tiger is not available

model = YoloObjectDetector('../darknet/darknet/cfg/yolov3.cfg', '../darknet/darknet/yolov3.weights')

runModelMetrics(model, images, yolo_labels, yolo_expected_classid, 'YOLO')

print()
model = ObjectDetector('ssd/saved_model')
ssd_expected_classid = 388 # Tiger
if 'amur' not in folder:
    ssd_expected_classid = 448

ssd_labels = loadSsdLabels('ssd/label_mapping.csv')

runModelMetrics(model, images, ssd_labels, ssd_expected_classid, 'MobileNet-SSD')

plt.show()
