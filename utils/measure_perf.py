from yolo_object_detection import YoloObjectDetector
from object_detection import ObjectDetector
import time
import sys
import numpy as np
import os
import cv2

args = sys.argv

if len(args) < 3:
    print('Usage: cmd <model config> <model weights> [num files]')
    exit()

n = 10
if len(args) > 3:
    n = int(args[3])

folder = 'amur/detection_train/trainval'
images = ['{}/{}'.format(folder, x) for x in os.listdir(folder)]

imagelist = np.random.choice(images, n, replace=False)

print('Measuring performance over {} images...'.format(len(images)))

config = args[1]
weights = args[2]

#model = YoloObjectDetector(config, weights, '../darknet/darknet/custom_data/detector.data')
#model.loadModel()

model = ObjectDetector(config)
model.loadModel()


print('Warming up...')
#model.getBoundingBoxes(images[0])
#model.getBoundingBoxes(images[1])
model.getBoundingBoxes(cv2.imread(images[0]))
model.getBoundingBoxes(cv2.imread(images[1]))
[cv2.imread(x) for x in imagelist]

print('Starting...')
times = []
for im in imagelist:
    #model.getBoundingBoxes(im)
    img = cv2.imread(im)
    st = time.time()
    model.getBoundingBoxes(img)
    times.append(time.time() - st)

print('Total time taken: {:.4f}s, Average time per detection: {:.4f}s, Min time: {:.4f}s, Max time: {:.4f}s'.format(sum(times), (sum(times))/n, min(times), max(times)))
