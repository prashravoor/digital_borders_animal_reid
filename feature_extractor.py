import onnxruntime as rt
import cv2
import numpy as np
import torch
from torchvision import transforms

class FeatureExtractor:
    def __init__(self, modelpath, width=256, height=256, threads=64):
        self.modelpath = modelpath
        self.INPUT_WIDTH = width
        self.INPUT_HEIGHT = height
        self.THREADS = threads
        self.session = None
        self.input_name = None

    def loadModel(self):
        sess_options = rt.SessionOptions()

        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        sess_options.intra_op_num_threads = self.THREADS
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # To enable model serialization and store the optimized graph to desired location.
        sess_options.optimized_model_filepath = self.modelpath 
        self.session = rt.InferenceSession(self.modelpath, sess_options)

        self.input_name = self.session.get_inputs()[0].name

    def extract(self, img):
        img = img[:,:,(2,1,0)] # Convert BGR to RGB
        img = cv2.resize(img, (self.INPUT_HEIGHT, self.INPUT_WIDTH))
        transform = transforms.Compose([ # Subtract mean and std-dev, standard procedure
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                       ])
        img = transform(img).numpy()
        img = np.expand_dims(img, axis=0)
        res = self.session.run(None, {self.input_name: img})
        return res[0].flatten()

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
