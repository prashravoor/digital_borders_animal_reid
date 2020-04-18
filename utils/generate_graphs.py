import numpy as np
import sys
from test_svm_model import *
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from joblib import dump,load

AccRecord = namedtuple('AccRecord', 'imType numIds samplesPerId numImages accuracy')

def get_num_ids(folder):
    with open(os.path.join(folder, 'normalized_class_mapping.txt')) as f:
        mapping = {x.split('\t')[0].strip(): int(x.split('\t')[1].strip()) for x in f.readlines()}

    rev_map = {}
    for k,v in mapping.items():
        if not v in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)

    return len(rev_map)

def getAccForFolderVaryNumIds(folder, model, imType, samples_per_id, detector, start=5, end=75):
    ids_num = get_num_ids(folder)
    print('Found {} ids for Type {}'.format(ids_num, imType))

    acc_list = [] 
    max_images = 1000
    minSamplesPerId=3
    for i in range(start,end+1,5):
        num_ids = int(np.ceil((i/100.0) * ids_num))
        correct,count,times = find_acc(model, folder, max_images, num_ids, minSamplesPerId, samples_per_id, detector)
        acc_list.append(AccRecord(imType, num_ids, samples_per_id, count, correct/count))

    return acc_list

def getAccForFolderVarySamplesPerId(folder, model, imType, num_ids, detector, start=1, end=20):
    ids_num = get_num_ids(folder)
    print('Found {} ids for Type {}'.format(ids_num, imType))

    acc_list = [] 
    max_images = 1000
    minSamplesPerId=1
    for i in range(start,end+1,2):
        correct,count,times = find_acc(model, folder, max_images, num_ids, minSamplesPerId, i, detector)
        acc_list.append(AccRecord(imType, num_ids, i, count, correct/count))

    return acc_list


def plot_bar(result, xvalues, xlabel, title):
    yvalues = [x.accuracy for x in result]

    _ = plt.figure()
    ax = plt.subplot()
    ax.bar(x=xvalues, height=yvalues)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)

if 'plotonly' in sys.argv:
    result_jag_s = load('result_jag_s')
    result_tig_s = load('result_tig_s')
    result_jag = load('result_jag')
    result_tig = load('result_tig')
    num_ids = 15
    samples_per_id = 5 
    plot_bar(result_jag_s, [x.samplesPerId for x in result_jag_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Jaguar', num_ids))
    plot_bar(result_jag, [x.numIds for x in result_jag], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Samples Per Id = {})'.format('Jaguar', samples_per_id))
    plot_bar(result_tig_s, [x.samplesPerId for x in result_tig_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Tiger', num_ids))
    plot_bar(result_tig, [x.samplesPerId for x in result_tig], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Tiger', samples_per_id))
    plt.show()

print('Loading object detector...')
det = ObjectDetector('ssd/saved_model')
det.loadModel()

print('Warming up...')
# Warmup
#det.getBoundingBoxes('amur_small/002019.jpg')
#det.getBoundingBoxes('amur_small/002019.jpg')

jag_model = 'jaguar-alexnet-fc7-linear-pca.model'
amur_model = 'amur-alexnet-relu6-linear-pca.model'
elp_modl = 'elp-alexnet-fc7-linear-pca.model'

jag_img = 'jaguars/reid'
amur_img = 'amur/plain_reid_train/train'
elp_img = 'ELPephants/images'

num_ids = 15
#result_jag_s = getAccForFolderVarySamplesPerId(jag_img, jag_model, 'Jaguar', num_ids, det, end=10)
#result_tig_s = getAccForFolderVarySamplesPerId(amur_img, amur_model, 'Tiger', num_ids, det)
result_elp_s = getAccForFolderVarySamplesPerId(elp_img, elp_model, 'Elephant', num_ids, det)

samples_per_id = 5 
#result_jag = getAccForFolderVaryNumIds(jag_img, jag_model, 'Jaguar', samples_per_id, det, end=55)
#result_tig = getAccForFolderVaryNumIds(amur_img, amur_model, 'Tiger', samples_per_id, det)
result_elp = getAccForFolderVaryNumIds(elp_img, elp_model, 'Elephant', samples_per_id, det)

#dump(result_jag_s, 'results_jag_s')
#dump(result_jag, 'results_jag')
#dump(result_tig_s, 'results_tig_s')
#dump(result_tig, 'results_tig')
dump(result_elp_s, 'results_elp_s')
dump(result_elp, 'results_elp')

#plot_bar(result_jag_s, [x.samplesPerId for x in result_jag_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Jaguar', num_ids))
#plot_bar(result_jag, [x.numIds for x in result_jag], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Samples Per Id = {})'.format('Jaguar', samples_per_id))
#plot_bar(result_tig_s, [x.samplesPerId for x in result_tig_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Tiger', num_ids))
#plot_bar(result_tig, [x.numIds for x in result_tig], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Tiger', samples_per_id))
plot_bar(result_elp_s, [x.samplesPerId for x in result_elp_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Elephant', num_ids))
plot_bar(result_elp, [x.numIds for x in result_elp], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Elephant', samples_per_id))
plt.show()
