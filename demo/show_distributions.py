import os
import matplotlib.pyplot as plt

def getDistributions(path, anType):
    with open(os.path.join(path, 'class_mapping.txt')) as f:
        mapping = {x.split('\t')[0].strip(): x.split('\t')[1].strip() for x in f.readlines()}
    
    rev_map = {}
    for k,v in mapping.items():
        if not v in rev_map:
            rev_map[v] = []
        rev_map[v].append(k)
    
    vals = sorted([(k, len(v)) for k,v in rev_map.items()], key=lambda x: -x[1])
    #plt.bar([x for x,y in vals], [y for x,y in vals])
    plt.hist([y for x,y in vals], rwidth=0.9)
    plt.xlabel('Number of Images Per Individual')
    # plt.xticks([])
    plt.ylabel('Number of Individuals')
    plt.title('{} Dataset - Distribution of images per identifier.\nNumber of individuals: {}'.format(anType, len(vals)))
    plt.show()
    return vals
    
#getDistributions('amur/plain_reid_train/train', 'Tigers')
#getDistributions('jaguars/reid', 'Jaguars')
#getDistributions('ELPephants/images', 'Elephants')
getDistributions('J:\\Source\\facescrub\\FaceScrub\\cropped\\flat_faces', 'FaceScrub')