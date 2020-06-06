from scipy.io import loadmat
import sys
import numpy as np
import os

def parseMdFile(filename):
    with open(filename) as f:
        lines = [x.strip() for x in f.readlines()]

    # Format of lines: Filename <> Name <> ...
    # Collect names to a set, and map each name to single identity
    names = set([x.split()[3].strip() for x in lines])
    print('Found total of {} identities'.format(len(names)))

    idfile = os.path.join(os.path.dirname(filename), 'id_mapping.txt')
    if not os.path.exists(idfile):
        if 'czoo' in filename:
            counter = 100
        else:
            counter = 0
        idmapping = {}
        for n in names:
            idmapping[n] = counter
            counter += 1
            
        with open(idfile, 'w') as f:
            f.writelines('\n'.join(['{}\t{}'.format(k,v) for k,v in idmapping.items()]))
    else:
        print('Reading ids from existing file..')
        with open(idfile) as f:
            idmapping = {x.split('\t')[0].strip() : int(x.split('\t')[1].strip()) for x in f.readlines()}

    mapping = {x.split()[1].strip() : idmapping[x.split()[3].strip()] for x in lines}
    return mapping    

args = sys.argv
if len(args) < 3:
    print('Usage: cmd <Czoo / CTai directory> <splits file> [split number]')
    exit(1)

infolder = args[1]
splitsfile = args[2]

splits = loadmat(splitsfile)
splits = splits['dataset_splits'][0]

splitNum = np.random.randint(0, 5)
if len(args) > 3:
    try:
        splitNum = int(args[3])
    except:
        pass

split = splits[splitNum][0][0]
trainIndices = split[0]
testIndices = split[1]

trainIndices = [x[0]-1 for x in trainIndices] # array is 1-indexed
testIndices = [x[0]-1 for x in testIndices]

print('Loaded splits, using split number {} to generate data'.format(splitNum))
print('Total training images: {}, Test images: {}'.format(len(trainIndices), len(testIndices)))

trainoutfile = os.path.join(infolder, 'split_{}_train.txt'.format(splitNum))
testoutfile = os.path.join(infolder, 'split_{}_test.txt'.format(splitNum))

imagesfile = os.path.join(infolder, 'filelist_face_images.txt')

mdfile = os.path.join(infolder, 'annotations_czoo.txt')
if not os.path.exists(mdfile):
    mdfile = os.path.join(infolder, 'annotations_ctai.txt')
    
with open(imagesfile) as f:
    images = [x.strip() for x in f.readlines()]
    
print('Total {} images in file list'.format(len(images)))
mapping = parseMdFile(mdfile)
trainimages = ['{}\t{}'.format(os.path.realpath(os.path.join(infolder, images[x])), mapping[images[x]]) for x in trainIndices]
testimages = ['{}\t{}'.format(os.path.realpath(os.path.join(infolder, images[x])), mapping[images[x]]) for x in testIndices]

with open(trainoutfile, 'w') as f:
    f.writelines('\n'.join(trainimages))
    
with open(testoutfile, 'w') as f:
    f.writelines('\n'.join(testimages))