from object_detection import ObjectDetector
import time
import os
import sys
import cv2

args = sys.argv
if not len(args) == 2 and not len(args) == 4:
    print('Usage: {} <images folder> [lower] [upper]'.format(args[0]))
    exit(1)

folder = args[1]
imageFiles = ['{}/{}'.format(folder, x) for x in os.listdir(folder) if x.endswith('.jpg')]
n = min(100,len(imageFiles))

print('Loading Model...')
det = ObjectDetector('ssd/saved_model')
det.loadModel()

print('Warming up model...')
# Run a dummy test once for warmup
img = cv2.imread(imageFiles[0])
det.getBoundingBoxes(img)
det.getBoundingBoxes(img)

batchTimes = []
lower=10
upper=11

if len(args) == 4:
    lower = int(args[2])
    upper = int(args[3])

f = open('test_times.csv', 'w')
f.write('Batch Size,Total Time,Average Time\n')
for i in range(lower,upper,5):
    batchSize = i
    # Detect batchSize images at a time
    cur = 0
    print('Running inference {} images at a time ...'.format(batchSize))
    start = time.time()
    while cur < n:
        images = imageFiles[cur:min(n,cur+batchSize)]
        cur += batchSize
        det.getBatchDetectionResults([cv2.imread(x) for x in images])
    total = time.time() - start
    print('Total detection time for batchSize {}: {:.2f}s, Average: {:.2f}s'.format(i, total, total/n))
    f.write('{},{},{}\n'.format(i, total, total/n))

print('Completed...')
f.close()
