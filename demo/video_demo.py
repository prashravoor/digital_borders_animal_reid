from object_detection import ObjectDetector
from im_utils import *
from svm_identifier import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from im_utils import *
import sys
import os

if not len(sys.argv) == 2:
    print('Usage: {} <Video File>'.format(sys.argv[0]))
    exit(1)

video = sys.argv[1]
if not os.path.exists(video):
    print('Error! File {} does not exists'.format(video))
    exit(1)

print('Loading Object Detector...')
detector = ObjectDetector('ssd/saved_model')
detector.loadModel()
#detector.getBoundingBoxes(cv2.imread('amur_small/002019.jpg'))
#detector.getBoundingBoxes(cv2.imread('amur_small/002019.jpg'))

print('Loading Identifiers...')
tigers_identifier = SvmIdentifier('amur-alexnet-relu6-linear-pca.model', detector)
jaguars_identifier = SvmIdentifier('jaguar-alexnet-fc7-linear-pca.model', detector)
elephants_identifier = SvmIdentifier('elp-alexnet-fc7-linear-pca.model', detector)

tigers_identifier.loadModel()
jaguars_identifier.loadModel()
elephants_identifier.loadModel()

print('Warming Up...')
tigers_identifier.predictId('amur_small/002808.jpg')
elephants_identifier.predictId('373_Ariel II right_Oct 2004.jpg')
jaguars_identifier.predictId('jaguar_small/j157_5_0.jpg')

print('Starting Video Analysis...')

with open('ssd/label_mapping.csv') as f:
    mapping = {int(x.split(',')[0].strip()) : x.split(',')[1].strip() for x in f.readlines()}

vs = cv2.VideoCapture(video)
while True:
    ret, frame = vs.read()
    if not ret:
        break
     
    results = detector.getBoundingBoxes(frame)
    if len(results) == 0:
        continue
        
    detections = []
    for res in results:
        det_id = 'N/A'
        
        if jaguars_identifier.belongsToClass(res.classid):
            det_id = jaguars_identifier.getIdFromBoundingBox(frame, res.bounding_box)
        elif tigers_identifier.belongsToClass(res.classid):
            det_id = tigers_identifier.getIdFromBoundingBox(frame, res.bounding_box)
        elif elephants_identifier.belongsToClass(res.classid):
            det_id = elephants_identifier.getIdFromBoundingBox(frame, res.bounding_box)
        
        detections.append(DetectionResult(res.bounding_box, res.classid, det_id))
            
    for r in detections:
        label = '{}, ID: {}'.format(mapping[r.classid], r.identifier)
        image = drawBoundingBoxLabel(frame, label, r)
        cv2.imshow('Demo', image)
        cv2.waitKey(1)
cv2.destroyAllWindows()
