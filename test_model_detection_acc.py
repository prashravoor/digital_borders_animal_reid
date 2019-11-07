from object_detection import ObjectDetector
import os
import sys
from im_utils import iou
import json
import numpy as np
import cv2
import time

def func():
    for j in range(i,m):
        imageIndex = os.path.basename(imageFiles[j]).split('.')[0]
        ground_truth = box_map[imageIndex]
        result = results[j-i]

        if not len(result) == len(ground_truth):
            print('Incorrect number of detections for image {}, Actual: {}, Expected: {}'
                        .format(imageFiles[j], len(result), len(ground_truth)))
        for res in result:
            ap = 0.
            if not result.classid == expected_classid:
                print('Incorrect classification for {}, Actual class id: {}, Expected: {}'
                        .format(imageFiles[j], res.classid, expected_classid))


def find_best_matching_pairs(ground_truth, pred):
    pairs = []
    if len(ground_truth) == 0 or len(pred) == 0:
        return pairs

    if not len(ground_truth) == len(pred):
        print('Mismatched number of predictions: GT: {}, Pred: {}'
            .format(len(ground_truth), len(pred)))

    iou_table = np.zeros((len(ground_truth), len(pred)))
    for i in range(len(ground_truth)):
        for j in range(len(pred)):
            iou_table[i][j] = iou(ground_truth[i], pred[j])

    # print(iou_table)
    for _ in range(min(len(ground_truth), len(pred))):
        # Find max value in table, and set it to 0
        maxV = np.where(iou_table == iou_table.max())
        x,y = maxV[0][0], maxV[1][0]
        iou_table[x][y] = float('-inf')
        pairs.append([ground_truth[x], pred[y]])
    return pairs

def find_average_iou(box_pairs, gt_boxes):
    ap = .0

    if len(box_pairs) == 0:
        return ap

    for overlap in [.5, .75, .9]:
        correct = 0
        for pair in box_pairs:
            # print(pair, iou(pair[0], pair[1]))
            if iou(pair[0], pair[1]) >= overlap:
                correct += 1
        prec = correct/gt_boxes # Precision on correct detections
        ap += prec
    return ap/3. # Average precision over 3 different overlaps


args = sys.argv
if not len(args) == 3 and not len(args) == 4:
    print('Usage: {} <Images Folder> <Annotations File>'.format(args[0]))
    exit(1)

folder = args[1]

imageFiles = ['{}/{}'.format(folder, x.strip()) for x in os.listdir(folder) if x.endswith('.jpg')]

if len(args) == 4:
    n = min(int(args[3]), len(imageFiles))
else:
    n = len(imageFiles)

imageFiles = imageFiles[:n]

print('Running Detection on {} images'.format(n))

box_map = {}
with open(args[2]) as f:
    box_map = json.load(f)
    f.close()

print('Loading Object Detector...')
det = ObjectDetector('ssd/saved_model')
det.loadModel()

print('Warming up model...')
det.getBoundingBoxes(cv2.imread(imageFiles[0]))
det.getBoundingBoxes(cv2.imread(imageFiles[0]))

batchSize = 10
expected_classid = 388 # Only tigers
mAP = 0.
numBoxes = 0
wrong = 0
correct = 0
start = time.time()
for i in range(0,n,batchSize):
    m = min(n, i+batchSize)
    images = [cv2.imread(imageFiles[x]) for x in range(i,m)]

    results = det.getBatchDetectionResults(images)
    for j in range(i,m): # For each image in batch
        imageIndex = os.path.basename(imageFiles[j]).split('.')[0]
        ground_truth = box_map[imageIndex]
        result = results[j-i]
        if not len(result) == len(ground_truth):
            print('Incorrect number of detections for image {}, Actual: {}, Expected: {}'
                        .format(imageFiles[j], len(result), len(ground_truth)))
        for res in result:
            if not res.classid == expected_classid:
                print('Incorrect classification for {}, Actual class id: {}, Expected: {}'
                        .format(imageFiles[j], res.classid, expected_classid))
                wrong += 1
            else:
                correct += 1
        result = [x for x in result if x.classid == expected_classid]

        # Find best matching BBox-pairs
        # Each result has array of DetectionResult
        pred = [ [x.bounding_box.xmin,
                  x.bounding_box.ymin,
                  x.bounding_box.xmax,
                  x.bounding_box.ymax] for x in result]
        box_pairs = find_best_matching_pairs(ground_truth, pred)
        mAP += find_average_iou(box_pairs, len(ground_truth))
        numBoxes += len(ground_truth)

print()
print('------- Summary -------')
print('Total runtime for {} images: {:.2f}s'.format(n, (time.time() - start)))
print('Total Number of Boxes over {} images: {}, Predicted Boxes: {}, Correctly classified: {}, Accuracy: {:.2f}'
        .format(n, numBoxes, wrong+correct, correct, (float(correct)/numBoxes)))
print('Final mAP: {:.2f}'.format(mAP/n))
