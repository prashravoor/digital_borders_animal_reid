from train_validate_svm import *
from extract_features_and_store_to_mongo import *
import numpy as np
from db_interface import DbInterface,DbRecord
from joblib import dump, load
from sklearn.decomposition import IncrementalPCA
import sys
import os
from object_detection import ObjectDetector
import time
import argparse
from test_svm_model import *


print('Loading object detector...')
det = ObjectDetector('ssd/saved_model')
det.loadModel()

#correct,count,times,cl_times = find_acc('jaguar-alexnet-fc7-linear-pca.model', 'jaguars/reid', 100, None, 3, None, det)
#print('Jagaurs Test Accuracy: {:.4f}, out of {} images. Average Id Time: {:.3f}s, Average Classification Times: {:.3f}s'.format(correct/count, count, times/count, cl_times/count))

correct,count,times,cl_times = find_acc('amur-alexnet-relu6-linear-pca.model', 'amur/plain_reid_train/train', 100, None, 3, None, det)
print('Tigers Test Accuracy: {:.4f}, out of {} images. Average Id Time: {:.3f}s, Average Classification Times: {:.3f}s'.format(correct/count, count, times/count, cl_times/count))

#correct,count,times,cl_times = find_acc('elp-alexnet-fc7-linear-pca.model', 'ELPephants/images', 100, None, 3, None, det)
#print('Elephants Test Accuracy: {:.4f}, out of {} images'.format(correct/count, count, times/count, cl_times/count))

