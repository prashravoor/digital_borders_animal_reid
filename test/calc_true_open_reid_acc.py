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

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances on GPU
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    sample_1 = torch.tensor(sample_1).cuda()
    sample_2 = torch.tensor(sample_2).cuda()
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return (torch.sqrt(eps + torch.abs(distances_squared))).cpu().numpy()
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return ((eps + inner) ** (1. / norm)).cpu().numpy()
    
def getRandomClosedReidSplits(img_folder):
    # Closed set ReID

    images = [x for x in os.listdir(img_folder) if x.endswith('.jpg') or x.endswith('.png')]

    filename = 'class_mapping.txt'
    if not os.path.exists(filename):
        filename = 'normalized_class_mapping.txt'
    with open(os.path.join(img_folder, filename)) as f:
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
        # For each remaining identity, 25% is moved to qry, and rest to identity
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
    
def getRandomOpenReidSplits(img_folder):
    # Open set ReID

    images = [x for x in os.listdir(img_folder) if x.endswith('.jpg') or x.endswith('.png')]

    filename = 'class_mapping.txt'
    if not os.path.exists(filename):
        filename = 'normalized_class_mapping.txt'
    with open(os.path.join(img_folder, filename)) as f:
        mapping = {x.split('\t')[0].strip() : x.split('\t')[1].strip() for x in f.readlines()}

    mapping = {os.path.join(img_folder, k):v for k,v in mapping.items() if k in images}

    rev_map = dict()
    for k,v in mapping.items():
        if v not in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)

    rev_map = {k:v for k,v in rev_map.items() if len(v) > 2}
    numclasses = len(rev_map)

    x_gal_names = []
    y_gal = []
    x_qry_names = []
    y_qry = []

    # 5% of ids appear only in query, and not in gallery
    num_openids = int(np.ceil(0.05 * numclasses))
    openids = np.random.choice(list(rev_map.keys()), num_openids, replace=False)

    for id in openids:
        num = int(np.ceil(0.25 * len(rev_map[id])))
        use = np.random.choice(rev_map[id], num, replace=False)
        ids = [mapping[x] for x in use]
        x_qry_names.extend(use)
        y_qry.extend(ids)
        
    print('{} Identities used for open set, total {} images'.format(num_openids, len(x_qry_names)))

    for k,v in rev_map.items():
        # For each remaining identity, 25% is moved to qry, and rest to gallery
        if k in openids:
            continue
        
        num = int(np.ceil(0.25 * len(v)))
        qry = np.random.choice(v, num, replace=False)
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

    assert not num_ids == num_t_ids

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

def getAllCmc(x_gal, x_qry, y_gal, y_qry, threshold_dist, ranks):
    # Open Re-ID impl
    distmat = pdist(x_qry, x_gal)
    m, n = distmat.shape
    SENTINEL_CLASS = 9999
    y_gal_unique = set(y_gal)
    # Append dummy class at end of y_gal
    y_gal = np.append(y_gal, SENTINEL_CLASS)    

    all_cmc = []
    all_AP = []
    for x in range(m):
        distances = distmat[x]
        # Append Thresold distance
        distances = np.append(distances, threshold_dist)
        indices = np.argsort(distances)
        org_classes = y_gal[indices]
        org_class = y_qry[x]
        openreid = False
        if not org_class in y_gal_unique:
            org_class = SENTINEL_CLASS # Query Id does not appear in gallery
            openreid = True

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

def getMeanAndStd(distmat):
    m, n = distmat.shape
    min_dists = []
    for x in range(m):
        distances = distmat[x]
        indices = np.argsort(distances)
        min_dists.append(distances[indices[0]])
    return np.mean(min_dists), np.std(min_dists)
    
if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 4:
        print('Usage: cmd <model path> <image train folder> <image test folder>')
        exit()
    
    manualSeed = 7

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    modelpath = args[1]
    img_train_folder = args[2]
    img_test_folder = args[3]

    model = loadModel(modelpath)

    x_train_gal_names, y_gal_train, x_train_qry_names, y_qry_train = getRandomClosedReidSplits(img_train_folder)
    x_train_gal, x_train_qry = extractFeatures(x_train_gal_names, x_train_qry_names, model)
    distmat_train = pdist(x_train_qry, x_train_gal)
    mean, std = getMeanAndStd(distmat_train)
    threshold_dist = mean + (2 * std)
    print('After loading Training Images, Mean distance is: {:.4f}, Standard deviation: {:.4f}, Threshold Distance for Open Re-Id: {:.4f}'.format(mean, std, threshold_dist))

    ranks = [1, 3, 5, 10, 20, 50]
    maxrank = max(ranks)
    mean_cmc = []
    all_map = []
    first = True
    for _ in range(3):
        x_gal_names, y_gal, x_qry_names, y_qry = getRandomOpenReidSplits(img_test_folder)
        x_gal, x_qry = extractFeatures(x_gal_names, x_qry_names, model)
        all_cmc, mAP = getAllCmc(x_gal, x_qry, y_gal, y_qry, threshold_dist, ranks)
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
    print('mAP: {:.4f}, Mean Accuracy: {}'.format(mAP, 
        ','.join(['{:.4f} +- {:.4f}'.format(means[x-1], stds[x-1]) for x in ranks])))