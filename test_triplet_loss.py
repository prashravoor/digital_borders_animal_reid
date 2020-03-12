import sys
from sys import exit
from feature_extractor import FeatureExtractor
import cv2
import os
import numpy as np
import time
from sklearn import svm
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

def showTsneGraph(x_train, y_train, title):
    fig, ax = plt.subplots()

    X_embedded = TSNE(n_components=2).fit_transform(x_train)
    x, y = X_embedded[:,0], X_embedded[:,1]
    colors = plt.cm.rainbow(np.linspace(0,1,10))
    sc = ax.scatter(x, y, c=[int(x) for x in y_train], cmap=matplotlib.colors.ListedColormap(colors))
    plt.colorbar(sc)
    ax.set_title(title)

def createSvmModel(kernelType='linear'):
    if 'poly' == kernelType:
        return svm.SVC(kernel=kernelType, gamma='scale', degree=2)
    return svm.SVC(kernel=kernelType, gamma='scale')

def shuffle(x,y):
    c = list(zip(x,y))
    np.random.shuffle(c)
    return zip(*c)

def getTrainValData(folder, md_file):
    with open(md_file) as f:
        mapping = {x.split()[0].strip() : x.split()[1].strip() for x in f.readlines()}

    images = [x for x in os.listdir(folder) if x.endswith('.jpg')]
    mapping = {os.path.join(folder, k):v for k,v in mapping.items() if k in images}

    rev_map = dict()
    for k,v in mapping.items():
        if v not in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)

    rev_map = {k:v for k,v in rev_map.items() if len(v) > 1}

    x_train_names = []
    y_train = []
    x_val_names = []
    y_val = []

    for k,v in rev_map.items():
        # 30% split of each class to training and test
        num_el = int(np.ceil(len(v) * 0.3))
        val = np.random.choice(v, num_el, replace=False)
        train = [x for x in v if x not in val]
        x_train_names.extend(train)
        x_val_names.extend(val)
        y_train.extend([k for _ in range(len(train))])
        y_val.extend([k for _ in range(len(val))])
        assert len(x_train_names) == len(y_train), 'Mismatch in Training samples and labels!'
        assert len(x_val_names) == len(y_val), 'Mismatch in Validation Samples and labels!'

    x_train_names, y_train = shuffle(x_train_names, y_train)
    x_val_names, y_val = shuffle(x_val_names, y_val)

    # Verify correctness of data
    for index in range(len(x_val_names)):
        assert mapping[x_val_names[index]] == y_val[index], 'Mismatch in labels: {} vs {}'.format(y_val[index], mapping[x_val_names[index]])
        
    for index in range(len(x_train_names)):
        assert mapping[x_train_names[index]] == y_train[index], 'Mismatch in labels: {} vs {}'.format(y_train[index], mapping[x_train_names[index]])

    return x_train_names, y_train, x_val_names, y_val

def mapLabelsToContinousInts(y_train, y_val):
# Map to congiguous identities
    num_ids = len(set(y_train))
    num_t_ids = len(set(y_val))

    assert num_ids == num_t_ids, 'Non-overlapping labels in training and val set'

    id_map = dict()
    counter = 0
    for id in y_train:
        if not id in id_map:
            id_map[id] = counter
            counter += 1

    y_train = [id_map[x] for x in y_train]
    y_val = [id_map[x] for x in y_val]
    return y_train, y_val

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 3:
        print('Usage: cmd imagefolder modelpath')
        exit()

    folder = args[1]
    modelpath = args[2]

    md_file = os.path.join(folder, 'normalized_class_mapping.txt')
    if not os.path.exists(md_file):
        md_file = os.path.join(folder, 'class_mapping.txt')
        if not os.path.exists(md_file):
            print('No metadata found in folder {}'.format(folder))
            exit()

    fe = FeatureExtractor(modelpath)
    print('Loading Model...')
    start = time.time()
    fe.loadModel()
    print('Warming up model...')
    fe.extract(cv2.imread('amur_small/002019.jpg'))
    print('Total time to load model: {:.4f}s'.format(time.time() - start))

    x_train_names, y_train, x_val_names, y_val = getTrainValData(folder, md_file)
    print('Train classes: {}, Valid Classes: {}'.format(len(set(y_train)), len(set(y_val))))
    print('Total Train images: {}, Total validation images: {}'.format(len(x_train_names), len(x_val_names)))

    print('Extracting features...This could take a while...')
    start = time.time()
# Extract feature for each image in training set
    x_train = np.asarray([fe.extract(cv2.imread(x)) for x in x_train_names])
    x_val = np.asarray([fe.extract(cv2.imread(x)) for x in x_val_names])
    total_time = time.time() - start

    print('Total Feature extraction time: {:.4f}s, Average time per feature extraction: {:.4f}s'.format(total_time, total_time / (len(x_train) + len(x_val))))

    y_train, y_val = mapLabelsToContinousInts(y_train, y_val)

    print('Fitting SVM...')
    start = time.time()
    svmModel = createSvmModel()
    svmModel = svmModel.fit(x_train, y_train)
    print('Model Fit time: {:.4f}s'.format(time.time() - start))

    print('Testing Accuracies...')

    start = time.time()
    y_pred = svmModel.predict(x_train)
    acc = metrics.accuracy_score(y_train, y_pred)
    print('Training Accuracy: {:.4f}'.format(acc))

    y_pred = svmModel.predict(x_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    print('Validation Accuracy: {:.4f}'.format(acc))
    total_time = time.time() - start

    print('Total Prediction time for {} samples: {:.4f}s, Average prediction time: {:.4f}s'.format(len(x_train) + len(x_val), total_time, total_time / (len(x_train) + len(x_val))))

    print('Displaying TSNE Graphs...')
    showTsneGraph(x_train, y_train, 'Training Set')
    showTsneGraph(x_val, y_val, 'Validation Set')
    plt.show()
