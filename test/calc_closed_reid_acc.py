import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.models as models
import cv2
from torchvision import datasets, transforms
import time
import os
import numpy as np
from scipy.spatial import distance_matrix
import sys
sys.path.insert(0, '../reid-strong-baseline')
from modeling import Baseline
import random

def loadModel(modelpath):
    dummy = 10
    if 'densenet' in modelpath:
        mdl = Baseline(dummy, 1, modelpath, 'bnneck', 'after', 'densenet', 'self')
    else:
        mdl = Baseline(dummy, 1, modelpath, 'bnneck', 'after', 'resnet50', 'self')
    mdl.load_param(modelpath)

    model = nn.DataParallel(mdl)
    model = model.to('cuda')
    model.eval()
    return model
    
def getRandomClosedReidSplits(img_folder):
    # Closed set ReID

    images = [x for x in os.listdir(img_folder) if x.endswith('.jpg') or x.endswith('.png')]

    with open(os.path.join(img_folder, 'class_mapping.txt')) as f:
        mapping = {x.split('\t')[0].strip() : x.split('\t')[1].strip() for x in f.readlines()}

    mapping = {os.path.join(img_folder, k):v for k,v in mapping.items() if k in images}

    rev_map = dict()
    for k,v in mapping.items():
        if v not in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)

    rev_map = {k:v for k,v in rev_map.items() if len(v) > 1}
    numclasses = len(rev_map)

    x_gal_names = []
    y_gal = []
    x_qry_names = []
    y_qry = []

    num_ids = len(rev_map)

    for k,v in rev_map.items():
        # For each remaining identity, one is moved to qry, and rest to identity
        #qry = np.random.choice(v, 1)[0]
        #x_qry_names.append(qry)
        #y_qry.append(mapping[qry])
        #gal = [x for x in v if not x == qry]
        
        #ids = [mapping[x] for x in gal]
        #x_gal_names.extend(gal)
        #y_gal.extend(ids)
        
        n = int(np.ceil(len(v) * .25))
        qry = np.random.choice(v, n)
        x_qry_names.extend(qry)
        y_qry.extend([mapping[x] for x in qry])
        gal = [x for x in v if not x in qry]
        
        ids = [mapping[x] for x in gal]
        x_gal_names.extend(gal)
        y_gal.extend(ids)

    def shuffle(x,y):
        c = list(zip(x,y))
        np.random.shuffle(c)
        return zip(*c)

    # Verify correctness of data
    for index in range(len(x_qry_names)):
        assert mapping[x_qry_names[index]] == y_qry[index], '{} != {}'.format(mapping[x_qry_names[index]],y_qry[index])
        
    for index in range(len(x_gal_names)):
        assert mapping[x_gal_names[index]] == y_gal[index]
        
    # Map to congiguous identities
    num_ids = len(set(y_gal))
    num_t_ids = len(set(y_qry))

    assert num_ids == num_t_ids

    id_map = dict()
    counter = 0
    for id in y_qry:
        if not id in id_map:
            id_map[id] = counter
            counter += 1

    rev_id_map = {v:k for k,v in id_map.items()}
            
    y_gal_old, y_qry_old = y_gal, y_qry
    y_gal = np.array([id_map[x] for x in y_gal])
    y_qry = np.array([id_map[x] for x in y_qry])

    for i in range(len(y_gal)):
        assert y_gal_old[i] == rev_id_map[y_gal[i]]

    for i in range(len(y_qry)):
        assert y_qry_old[i] == rev_id_map[y_qry[i]]

    x_gal_names, y_gal = shuffle(x_gal_names, y_gal)
    x_qry_names, y_qry = shuffle(x_qry_names, y_qry)
    y_gal = np.array(y_gal)
    y_qry = np.array(y_qry)

    #print('{} Identities used for closed set, total {} images'.format(num_openids, len(x_qry_names)))
    print('Train classes: {}, Valid Classes: {}'.format(len(set(y_gal)), len(set(y_qry))))
    print('Total Train images: {}, Total validation images: {}'.format(len(x_gal_names), len(x_qry_names)))
    return x_gal_names, y_gal, x_qry_names, y_qry
    
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
        return res.cpu().detach().numpy().flatten()
        
def extractFeatures(x_gal_names, x_qry_names, model):
    # Extract feature for each image in training set
    start = time.time()
    x_gal = np.asarray([extract_feature(cv2.imread(x), model) for x in x_gal_names])
    x_qry = np.asarray([extract_feature(cv2.imread(x), model) for x in x_qry_names])
    total_time = time.time() - start
    print('Total FE time: {:.4f}s, Average time per image: {:.4f}s'.format(
        total_time, total_time/(len(x_gal) + len(x_qry))))
        
    return x_gal, x_qry

def getAllCmc(x_gal, x_qry, y_gal, y_qry, ranks):
    distmat = distance_matrix(x_qry, x_gal)
    m, n = distmat.shape

    all_cmc = []
    all_AP = []
    for x in range(m):
        distances = distmat[x]
        indices = np.argsort(distances)
        org_classes = y_gal[indices]
        org_class = y_qry[x]
        matches = (org_class == org_classes).astype(int)
        
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)
        
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / len(all_cmc)
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP
    
def getAverageCmcMap(modelpath, img_folder, ranks = [1, 3, 5, 10, 20, 50]):
    maxrank = max(ranks)
    mean_cmc = []
    all_map = []
    first = True
    for _ in range(3):
        x_gal_names, y_gal, x_qry_names, y_qry = getRandomClosedReidSplits(img_folder)
        model = loadModel(modelpath)
        x_gal, x_qry = extractFeatures(x_gal_names, x_qry_names, model)
        all_cmc, mAP = getAllCmc(x_gal, x_qry, y_gal, y_qry, ranks)
        mean_cmc.append(all_cmc[:maxrank])
        all_map.append(mAP)
        
        if first:
            print(','.join(['Rank-{}'.format(x) for x in ranks]))
            first = False
        print(','.join([str(all_cmc[r-1]) for r in ranks]))
        
    mean_cmc = np.array(mean_cmc)
    means = np.mean(mean_cmc, axis=0)
    stds = np.std(mean_cmc, axis=0)
    mAP = np.mean(all_map)        
    
    return mAP, means, stds
        
if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 3:
        print('Usage: cmd <model path> <image folder>')
        exit()
    
    manualSeed = 42

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    modelpath = args[1]
    img_folder = args[2]
    
    ranks = [1, 3, 5, 10, 20, 50]
    mAP, means, stds = getAverageCmcMap(modelpath, img_folder, ranks)
    print('mAP: {:.4f}, Mean Accuracy: {}'.format(mAP, 
        ','.join(['{:.4f} +- {:.4f}'.format(means[x-1], stds[x-1]) for x in ranks])))
    
    