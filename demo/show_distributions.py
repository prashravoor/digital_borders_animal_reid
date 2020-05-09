import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib

def getDetails(path):
    filename = 'class_mapping.txt'
    
    if not os.path.exists(os.path.join(path, filename)):
        filename = 'normalized_class_mapping.txt'
    with open(os.path.join(path, filename)) as f:
        mapping = {x.split('\t')[0].strip(): x.split('\t')[1].strip() for x in f.readlines()}
    
    rev_map = {}
    for k,v in mapping.items():
        if not v in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)
    
    return rev_map

def getDistributions(path1, path2, anType):

    p1 = getDetails(path1)
    p2 = getDetails(path2)
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26}
    matplotlib.rc('font', **font)
    
    rev_map = p1
    for k,v in p2.items():
        if not k in rev_map:
            rev_map[k] = v
        else:
            rev_map[k].append(v)
    
    vals = sorted([(k, len(v)) for k,v in rev_map.items()], key=lambda x: -x[1])
        
    num_ids = len(rev_map)
    mean = np.mean([x[1] for x in vals])
    median = np.median([x[1] for x in vals])
    
    _,ax = plt.subplots(1,1)
    #plt.bar([x for x,y in vals], [y for x,y in vals])
    #plt.hist([y for x,y in vals], rwidth=0.9)
    #plt.xlabel('Number of Images Per Individual')
    # plt.xticks([])
    #plt.ylabel('Number of Individuals')
    #plt.title('{} Dataset - Distribution of images per identifier'.format(anType))
    
    #plt.text(200, 20, 'Number of individuals: {}\nMean samples per individual: {:.2f}\nMedian of samples per individual: {}'.format(num_ids, mean, median), style='normal', horizontalalignment='center', verticalalignment='center', transform=plt.transAxes, bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    
    #ax.bar([x for x,y in vals], [y for x,y in vals])
    ax.hist([y for x,y in vals], rwidth=0.9)
    ax.set_xlabel('Number of Images Per Individual')
    # plt.xticks([])
    ax.set_ylabel('Number of Individuals')
    ax.set_title('{} Dataset - Distribution of images per identifier'.format(anType))
    
    ax.text(0.95, 0.9, 'Number of individuals: {}\nMean samples per individual: {:.2f}\nMedian of samples per individual: {}'.format(num_ids, mean, median), style='normal', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
    
    plt.show()
    return vals

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('Usage: cmd <folder 1> <folder 2> <dataset name>')
        exit()
    #getDistributions('amur/plain_reid_train/train', 'Tigers')
    #getDistributions('jaguars/reid', 'Jaguars')
    #getDistributions('ELPephants/images', 'Elephants')
    #getDistributions('J:\\Source\\facescrub\\FaceScrub\\cropped\\flat_faces', 'FaceScrub')
    #getDistributions('J:\\Source\\facescrub\\FaceScrub\\cropped\\flat_faces', 'FaceScrub')
    getDistributions(args[1], args[2], args[3])