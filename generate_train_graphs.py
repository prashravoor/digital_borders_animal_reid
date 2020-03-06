import numpy as np
import sys
from test_svm_model import *
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from joblib import dump,load
from svm_identifier import *
import time
from train_validate_svm import *


AccRecord = namedtuple('AccRecord', 'imType numIds samplesPerId numImages accuracy')

def get_num_ids(folder):
    with open(os.path.join(folder, 'normalized_class_mapping.txt')) as f:
        mapping = {x.split('\t')[0].strip(): int(x.split('\t')[1].strip()) for x in f.readlines()}

    rev_map = {}
    for k,v in mapping.items():
        if not v in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)

    return len(rev_map),rev_map

def out_train_svm_for_layer(model, folder, num_ids, samples_per_id, detector):
    svmid = SvmIdentifier(model, detector)
    svmid.loadModel()

    db = DbInterface().getDB(svmid.dsName + '_' + svmid.modelName)
    recids = list(db.getRecordIds(svmid.layer))
    recs = [db.getRecord(svmid.layer, x) for x in recids]

    id_map = {}
    for rec in recs:
        if not rec.animalId in id_map:
            id_map[rec.animalId] = []
        id_map[rec.animalId].append(rec.imageFile)

    id_map = [(k,v) for k,v in id_map.items() if len(v) >= samples_per_id]
    # Modify list in place, randomly select samples_per_id images from each list
    for i in range(len(id_map)):
        id_map[i] = (id_map[i][0], np.random.choice(id_map[i][1], samples_per_id, replace=False))

    if len(id_map) < num_ids:
        print('Insufficient samples found for numIds = {}, samples per Id = {} for DS: {}, Model: {}'.format(num_ids, samples_per_id, svmid.dsName, svmid.modelName))
        return AccRecord(svmid.dsName, num_ids, samples_per_id, 0, 0)

    pick = np.random.choice(len(id_map), num_ids, replace=False)
    id_map = [id_map[x] for x in pick]

    imageFiles = []
    for x in id_map:
        imageFiles.extend(x[1])

    features = [x.feature for x in recs if x.imageFile in imageFiles]
    target = [x.animalId for x in recs if x.imageFile in imageFiles]
    print('Fitting {} features, for numIds = {}, samples per id = {}'.format(len(features), num_ids, samples_per_id))

    start = time.time()
    samples,pca_model = transformFeature(np.array(features), 'pca')
    X_train_indices, y_train, X_test_indices, y_test = train_val_split_closedset(samples, target, min_entries=1)
    shuffle(X_train_indices, y_train)
    shuffle(X_test_indices, y_test)
    x_train = [samples[x] for x in X_train_indices]
    x_test = [samples[x] for x in X_test_indices]

    svmModel = createSvmModel(svmid.kernelType)
    svmModel.fit(x_train, y_train)

    y_pred = svmModel.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    end = time.time()
    print('Total Fit + Predict Time: {:.4f}s'.format(end - start))

    return AccRecord(svmid.dsName, num_ids, samples_per_id, len(x_train)+len(x_test), acc)

def getAccForFolderVaryNumIds(folder, model, imType, samples_per_id, detector, start=5, end=75):
    ids_num,_ = get_num_ids(folder)
    acc_list = [] 
    for i in range(start,end+1,5):
        num_ids = int(np.ceil((i/100.0) * ids_num))
        acc = out_train_svm_for_layer(model, folder, num_ids, samples_per_id, detector)
        acc_list.append(acc)

    return acc_list

def getAccForFolderVarySamplesPerId(folder, model, imType, num_ids, detector, start=2, end=20):
    acc_list = [] 
    for i in range(start,end+1,2):
        acc = out_train_svm_for_layer(model, folder, num_ids, i, detector)
        acc_list.append(acc)

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

jag_model = 'jaguar-alexnet-fc7-linear-pca.model'
amur_model = 'amur-alexnet-relu6-linear-pca.model'
elp_model = 'elp-alexnet-fc7-linear-pca.model'

jag_img = 'jaguars/reid'
amur_img = 'amur/plain_reid_train/train'
elp_img = 'ELPephants/images'

num_ids = 15
result_jag_s = getAccForFolderVarySamplesPerId(jag_img, jag_model, 'Jaguar', num_ids, det, end=10)
dump(result_jag_s, 'results_jag_s')

result_tig_s = getAccForFolderVarySamplesPerId(amur_img, amur_model, 'Tiger', num_ids, det, end=8)
dump(result_tig_s, 'results_tig_s')

#result_elp_s = getAccForFolderVarySamplesPerId(elp_img, elp_model, 'Elephant', num_ids, det, start=2, end=4)
#dump(result_elp_s, 'results_elp_s')

samples_per_id = 5
result_jag = getAccForFolderVaryNumIds(jag_img, jag_model, 'Jaguar', samples_per_id, det, end=55)
dump(result_jag, 'results_jag')

result_tig = getAccForFolderVaryNumIds(amur_img, amur_model, 'Tiger', samples_per_id, det)
dump(result_tig, 'results_tig')

#result_elp = getAccForFolderVaryNumIds(elp_img, elp_model, 'Elephant', 3, det)
#dump(result_elp, 'results_elp')

plot_bar(result_jag_s, [x.samplesPerId for x in result_jag_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Jaguar', num_ids))
plot_bar(result_jag, [x.numIds for x in result_jag], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Samples Per Id = {})'.format('Jaguar', samples_per_id))
plot_bar(result_tig_s, [x.samplesPerId for x in result_tig_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Tiger', num_ids))
plot_bar(result_tig, [x.numIds for x in result_tig], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Samples per Id = {})'.format('Tiger', samples_per_id))
#plot_bar(result_elp_s, [x.samplesPerId for x in result_elp_s], 'Number of samples per Id', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Elephant', num_ids))
#plot_bar(result_elp, [x.numIds for x in result_elp], 'Number of Ids', 'SVM Classification accuracy for {}\n(Number of Identities = {})'.format('Elephant', samples_per_id))
plt.show()
