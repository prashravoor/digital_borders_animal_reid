import cv2
import cv2.dnn as dnn
import numpy as np
from collections import namedtuple

class YoloBoundingBox:
    def __init__(self, x, y, w, h, width, height):
        left_x = x - w/2.0
        left_y = y - h/2.0
        w = int(w * width)
        h = int(h * height)
        self.top_left = tuple( [int(left_x * width), int(left_y * height)] )
        self.xmin, self.ymin = self.top_left
        self.bottom_right = tuple( [self.top_left[0] + w, self.top_left[1] + h] )
        self.xmax, self.ymax = self.bottom_right

    def __repr__(self):
        return 'TopLeft: {}, BottomRight: {}'.format(self.top_left, self.bottom_right)

YoloDetectionResult = namedtuple('YoloDetectionResult', 'bounding_box confidence classid')

class YoloObjectDetector:
    def __init__(self, configFile, weightsFile, iou_threshold=0.5, nms_threshold=0.45):
        self.config = configFile
        self.weights = weightsFile
        self.network = None
        self.INPUT_WIDTH = 416
        self.INPUT_HEIGHT = 416 
        self.IOU_THRESHOLD = iou_threshold
        self.NMS_THRESHOLD = nms_threshold

    def loadModel(self):
        self.network = dnn.readNetFromDarknet(self.config, self.weights)

    def _getOutputLayers(self):
        if self.network:
            names = self.network.getLayerNames()
            return [names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
        return None

    def getBoundingBoxes(self, image):
        if not self.network:
            print('Model not laded!')
            return None

        width, height = image.shape[1], image.shape[0]
        image = cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))

        scale = 1/255.0
        blob = dnn.blobFromImage(image, scale, (self.INPUT_WIDTH, self.INPUT_HEIGHT), (0,0,0), swapRB=True, crop=False)
        self.network.setInput(blob)

        outputs = self.network.forward(self._getOutputLayers())
        boxes = []
        confidences = []
        class_ids = []
        for out in outputs:
            for detection in out:
                # Each detection has 85 fields - First 4 are the bounding box,
                # 5 to 85 contain scores for each of the 80 classes
                boxes.append(detection[:4])
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidences.append(float(scores[class_id]))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.IOU_THRESHOLD, self.NMS_THRESHOLD) 

        return [YoloDetectionResult(YoloBoundingBox(*boxes[i[0]], width, height),
                                    confidences[i[0]],
                                    class_ids[i[0]]) 
                                    for i in indices]

if __name__ == '__main__':
    import sys
    import time
    import matplotlib.pyplot as plt
    from im_utils import drawBoundingBoxWithLabel

    args = sys.argv
    if not len(args) == 2:
        print('Usage: {} <image file>'.format(args[0]))
        exit(1)

    def loadLabels(file):
        with open(file) as f:
            lines = [line.strip() for line in f.readlines()]
            f.close()

        return lines

    print('Loading Model...')
    det = YoloObjectDetector('../darknet/darknet/cfg/yolov3.cfg', '../darknet/darknet/yolov3.weights')
    det.loadModel()

    print('Reading Image...')
    image = cv2.imread(args[1])

    print('Running inference...')
    result = det.getBoundingBoxes(image)
    print('Found {} bounding boxes'.format(len(result)))

    print('Loading Labels...')
    labels = loadLabels('../darknet/darknet/data/coco.names')

    print('Drawing bounding boxes on image...')
    for res in result:
        image = drawBoundingBoxWithLabel(image, res, labels)

    plt.imshow(image)
    plt.show()
