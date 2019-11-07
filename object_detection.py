import tensorflow as tf
import cv2
import numpy as np
from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', 'ymin xmin ymax xmax')
DetectionResult = namedtuple('DetectionResult', 'bounding_box confidence classid')

class ObjectDetector:
    def __init__(self, model_path, image_width=300, image_height=300):
        self.model_path = model_path
        # Required width, height for MobileNetSSDV2. Change as needed
        self.INPUT_WIDTH = image_width 
        self.INPUT_HEIGHT = image_height
        self.model = None
        self.MAX_BOXES_PER_IMAGE = 100
        self.IOU_THRESHOLD = 0.5
        self.SCORE_THRESHOLD = 0.1

    def loadModel(self):
        self.model = tf.saved_model.load(self.model_path, tags=None)
        self.model = self.model.signatures['serving_default']

    def getBoundingBoxes(self, image):
        '''
            Returns list of detection results
        '''
        '''
        # Resize image as per model need
        width = image.shape[1] 
        height = image.shape[0]
        image = cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))

        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = self.model(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy()
                             for key,value in output_dict.items()}
        # detection_classes should be ints.
        det_classes = output_dict['detection_classes'].astype(np.int64)

        det_boxes = output_dict['detection_boxes']
        det_scores = output_dict['detection_scores']
        indices = tf.image.non_max_suppression(det_boxes, det_scores,
                                               self.MAX_BOXES_PER_IMAGE,
                                               iou_threshold=self.IOU_THRESHOLD,
                                               score_threshold=self.SCORE_THRESHOLD)
        
        if len(indices) == 0:
            return [] 

        det_boxes = np.asarray([det_boxes[x] for x in indices])
        det_scores = np.asarray([det_scores[x] for x in indices])
        det_classes = np.asarray([det_classes[x] for x in indices])
        # Convert the detection result to correct scale as per input image
        det_boxes = (det_boxes * [height, width, height, width]).astype(int)

        detectionResults = [DetectionResult( BoundingBox(*tuple(det_boxes[x])),
                                             det_scores[x],
                                             det_classes[x])
                                             for x in range(len(indices))]
        return detectionResults 
        '''
        return self.getBatchDetectionResults([image])[0]

    def getBatchDetectionResults(self, images):
        origDims = [(image.shape[1], image.shape[0]) for image in images]

        images = [cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT)) for image in images]

        input_tensor = tf.stack([tf.convert_to_tensor(image) for image in images])

        output_dict = self.model(input_tensor)

        detectionResults = []
        det_classes = output_dict['detection_classes'].numpy().astype(int)
        det_boxes = output_dict['detection_boxes'].numpy()
        det_scores = output_dict['detection_scores'].numpy()
        for i in range(len(images)):
            boxes = det_boxes[i]
            indices = tf.image.non_max_suppression(boxes, det_scores[i],
                                               self.MAX_BOXES_PER_IMAGE,
                                               iou_threshold=self.IOU_THRESHOLD,
                                               score_threshold=self.SCORE_THRESHOLD)
            if len(indices) == 0:
                detectionResults.append([])
                continue

            boxes = np.asarray([boxes[x] for x in indices])
            scores = np.asarray([det_scores[i][x] for x in indices])
            classes = np.asarray([det_classes[i][x] for x in indices])
            w,h = origDims[i]
            boxes = (boxes * [h,w,h,w]).astype(int)
            detectionResults.append([DetectionResult(BoundingBox(*boxes[x]),
                                     scores[x], classes[x])
                                     for x in range(len(indices))])


        return detectionResults

if '__main__' == __name__:
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
            lines = [x.split(',') for x in f.readlines()]
            f.close()

        mapping = {}
        for line in lines:
            mapping[int(line[0])] = line[1].strip()

        return mapping

    print('Loading Model...')
    det = ObjectDetector('ssd/saved_model')
    det.loadModel()

    print('Reading Image...')
    image = cv2.imread(args[1])

    print('Running inference...')
    result = det.getBoundingBoxes(image)
    print('Found {} boxes: {}'.format(len(result), result))

    print('Loading Labels...')
    labels = loadLabels('ssd/label_mapping.csv')

    print('Drawing bounding boxes on image...')
    for res in result:
        image = drawBoundingBoxWithLabel(image, res, labels)
        plt.imshow(image)
    plt.show()
