import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

class FeatureExtractor:
    def __init__(self, modelpath, width=256, height=256):
        self.modelpath = modelpath
        self.INPUT_WIDTH = width
        self.INPUT_HEIGHT = height
        self.model = None

    def loadModel(self):
        self.model = tf.saved_model.load(self.modelpath)

    def preprocessImage(self, img):
        img = cv2.resize(img, (self.INPUT_HEIGHT, self.INPUT_WIDTH))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.rollaxis(img, 2, 0) # Channels first

        img = preprocess_input(img)
        return img

    def extractMultiple(self, images):
        images = np.array([self.preprocessImage(x) for x in images])
        res = self.model(tf.convert_to_tensor(images))
        return res.numpy()

    def extract(self, img):
        return self.extractMultiple([img])[0]

if __name__ == '__main__':
    import sys
    import time

    args = sys.argv

    if not len(args) == 3:
        print('Usage: cmd modelpath imagepath')
        exit()

    model = args[1]
    image = args[2]

    fe = FeatureExtractor(model)

    print('Loading Model...')
    start = time.time()
    fe.loadModel()
    print('Load Time: {:.4f}s'.format(time.time() - start))

    print('Warming up...')
    fe.extract(cv2.imread(image))

    start = time.time()
    res = fe.extract(cv2.imread(image))
    print('Total feature extraction time: {:.4f}s'.format(time.time() - start))
    print('Features: {}'.format(res))
