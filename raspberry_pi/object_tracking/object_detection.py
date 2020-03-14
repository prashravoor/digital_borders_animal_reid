import numpy as np
from tflite_runtime.interpreter import Interpreter
import cv2
from utils import BoundingBox, DetectionResult

class ObjectDetector:
    def __init__(self, modelpath, width=300, height=300, numthreads=4):
        self.modelpath = modelpath
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.interpreter = None
        self.MAX_BOXES_PER_IMAGE = 10
        self.IOU_THRESHOLD = 0.5
        self.SCORE_THRESHOLD = 0.5
        self.NUM_THREADS = numthreads

    def loadModel(self):
        self.interpreter = Interpreter(self.modelpath)
        self.interpreter.set_num_threads(self.NUM_THREADS)
        self.interpreter.allocate_tensors()
        self.INPUT_TENSOR_NAME = self.interpreter.get_input_details()[0]['index']
        output = self.interpreter.get_output_details()
        self.OUTPUT_TENSOR_BOXES = output[0]['index']
        self.OUTPUT_TENSOR_CLASSES = output[1]['index']
        self.OUTPUT_TENSOR_SCORES = output[2]['index']

    def getBoundingBoxes(self, image):
        image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        input_tensor = np.expand_dims(image, axis=0)
        self.interpreter.set_tensor(self.INPUT_TENSOR_NAME, input_tensor)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.OUTPUT_TENSOR_BOXES)[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.OUTPUT_TENSOR_CLASSES)[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.OUTPUT_TENSOR_SCORES)[0] # Confidence of detected objects

        # Filter low scoring boxes
        results = []
        for i in range(len(boxes)):
            if not scores[i] > self.SCORE_THRESHOLD:
                continue

            ymin = int(max(1,(boxes[i][0] * self.IMG_HEIGHT)))
            xmin = int(max(1,(boxes[i][1] * self.IMG_WIDTH)))
            ymax = int(min(self.IMG_HEIGHT,(boxes[i][2] * self.IMG_HEIGHT)))
            xmax = int(min(self.IMG_WIDTH,(boxes[i][3] * self.IMG_WIDTH)))

            results.append(DetectionResult(BoundingBox(ymin, xmin, ymax, xmax),
                                scores[i], classes[i]))

        return results

if __name__ == '__main__':
    import sys
    import time

    args = sys.argv
    if not len(args) == 3:
        print('Usage: cmd <Model> <Imange Path>')
        exit()

    model = args[1]
    img = args[2]

    image = cv2.imread(img)
    if image is None:
        print('Image {} not found'.format(img))

    print('Loading Model...')
    detector = ObjectDetector(model)
    detector.loadModel()

    start = time.time()
    results = detector.getBoundingBoxes(image)
    end = time.time()

    print('Results: {}, Time: {:.4f}s'.format(results, end - start))
