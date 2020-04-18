import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.models as models
import cv2
from torchvision import datasets, transforms
import time
from modeling import build_model
from config import cfg
import os
import numpy as np
import sys
from sklearn import svm
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

def createSvmModel(kernelType='linear'):
    if 'poly' == kernelType:
        return svm.SVC(kernel=kernelType, gamma='scale', degree=2)
    return svm.SVC(kernel=kernelType, gamma='scale')

def loadData(img_folder):
    # Read in all ELP images
    #img_folder = 'ELPephants/reid_faces_renamed/test'
    #img_folder = 'ELPephants/reid_faces_renamed/train/'
    #img_folder = 'jaguars/reid_strong_baseline/train'
    #img_folder = 'amur/open_reid/test/'
    images = [x for x in os.listdir(img_folder) if x.endswith('.jpg')]

    with open(os.path.join(img_folder, 'class_mapping.txt')) as f:
        mapping = {x.split('\t')[0].strip() : x.split('\t')[1].strip() for x in f.readlines()}

    mapping = {os.path.join(img_folder, k):v for k,v in mapping.items() if k in images}

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
        assert len(x_train_names) == len(y_train)
        assert len(x_val_names) == len(y_val)

    def shuffle(x,y):
        c = list(zip(x,y))
        np.random.shuffle(c)
        return zip(*c)

    # Verify correctness of data
    for index in range(len(x_val_names)):
        assert mapping[x_val_names[index]] == y_val[index], '{} != {}'.format(mapping[x_val_names[index]],y_val[index])
        
    for index in range(len(x_train_names)):
        assert mapping[x_train_names[index]] == y_train[index]
        
    # Map to congiguous identities
    num_ids = len(set(y_train))
    num_t_ids = len(set(y_val))

    assert num_ids == num_t_ids

    id_map = dict()
    counter = 0
    for id in y_train:
        if not id in id_map:
            id_map[id] = counter
            counter += 1

    rev_id_map = {v:k for k,v in id_map.items()}
            
    y_train_old, y_val_old = y_train, y_val
    y_train = np.array([id_map[x] for x in y_train])
    y_val = np.array([id_map[x] for x in y_val])

    for i in range(len(y_train)):
        assert y_train_old[i] == rev_id_map[y_train[i]]

    for i in range(len(y_val)):
        assert y_val_old[i] == rev_id_map[y_val[i]]

    x_train_names, y_train = shuffle(x_train_names, y_train)
    x_val_names, y_val = shuffle(x_val_names, y_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    print('Train classes: {}, Valid Classes: {}'.format(len(set(y_train)), len(set(y_val))))
    print('Total Train images: {}, Total validation images: {}'.format(len(x_train_names), len(x_val_names)))
    
    return x_train_names, y_train, x_val_names, y_val

def extract_feature(img, model):
    img = cv2.resize(img, (256,256))
    img = img[:,:,(2,1,0)]
    transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           ])
    img = transform(img)
    img = img.reshape((1, 3, 256,256))
    with torch.no_grad():
        model.eval()
        img = torch.autograd.Variable(img).cuda()
        res = model(img)
        #res = torch.sigmoid(res)
        return res.cpu().detach().numpy().flatten()

def extractFeatures(x_train_names, x_val_names, model):
    # Extract feature for each image in training set
    start = time.time()
    x_train = np.asarray([extract_feature(cv2.imread(x), model) for x in x_train_names])
    x_val = np.asarray([extract_feature(cv2.imread(x), model) for x in x_val_names])
    total_time = time.time() - start
    print('Total FE time: {:.4f}s, Average time per image: {:.4f}s'.format(
        total_time, total_time/(len(x_train) + len(x_val))))
        
    return x_train, x_val

def trainSvm(x_train, y_train):
    svmModel = createSvmModel()
    svmModel = svmModel.fit(x_train, y_train)
    return svmModel

def getAccuracy(x_train, y_train, svmModel):
    y_pred = svmModel.predict(x_train)
    acc = metrics.accuracy_score(y_train, y_pred)
    return acc

def createTsne(x_train, y_train, title):
    fig, ax = plt.subplots()

    X_embedded = TSNE(n_components=2).fit_transform(x_train)
    x, y = X_embedded[:,0], X_embedded[:,1]
    colors = plt.cm.rainbow(np.linspace(0,1,10))
    sc = ax.scatter(x, y, c=[int(x) for x in y_train], cmap=matplotlib.colors.ListedColormap(colors))
    plt.colorbar(sc)
    ax.set_title(title)
    ax.axis('off')
    
args = sys.argv
if not len(args) == 4:
    print('Usage: cmd <model> <in folder> <config>')
    exit()
    
modelpath = args[1]
infolder = args[2]
configfile = args[3]

cfg.merge_from_file(configfile)
cfg.freeze()

x_train_names, y_train, x_val_names, y_val = loadData(infolder)
num_classes = len(set(y_train))

model = build_model(cfg, num_classes)
model.load_param('../reid-strong-baseline/elp_test/densenet_model_120.pth')
model = nn.DataParallel(model)
model = model.to('cuda')
model.eval()

x_train, x_val = extractFeatures(x_train_names, x_val_names, model)
svmModel = trainSvm(x_train, y_train)
tr_acc = getAccuracy(x_train, y_train, svmModel)
print('Training Accuracy: {:.4f}'.format(tr_acc))

val_acc = getAccuracy(x_val, y_val, svmModel)
print('Validation Accuracy: {:.4f}'.format(val_acc))

createTsne(x_train, y_train, 'TSNE over Training Set')
createTsne(x_val, y_val, 'TSNE over Validation Set')
plt.show()