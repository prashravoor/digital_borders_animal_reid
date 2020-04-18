import os
import sys
import numpy as np
import shutil

args = sys.argv
if not len(args) == 3:
    print('Usage: cmd <in folder> <out folder>')
    exit()

infolder = args[1]
outfolder = args[2]

filename = os.path.join(infolder, 'normalized_class_mapping.txt')
if not os.path.exists(filename):
    filename = os.path.join(infolder, 'class_mapping.txt')
    if not os.path.exists(filename):
        print('Metdata file {} doesnt exist'.format(filename))
        exit()

with open(filename) as f:
    file_map = {x.split('\t')[0].strip() : x.split('\t')[1].strip() for x in f.readlines()}
    

images = [x for x in os.listdir(infolder) if x.endswith('.jpg')]
print('Metadata file had {} image names, Actual Images: {}'.format(len(file_map), len(images)))

file_map = {k:v for k,v in file_map.items() if k in images}
print('Filtered out non-existent image files, total number of images: {}'.format(len(file_map)))

rev_map = dict()
for k,v in file_map.items():
    if not v in rev_map:
        rev_map[v] = []
    rev_map[v].append(k)

# Remove identities with less than 3 images
rev_map = {k:v for k,v in rev_map.items() if len(v) > 2}
print('Total number of identities: {}'.format(len(rev_map)))

trainfolder = os.path.join(outfolder, 'train')
valfolder = os.path.join(outfolder, 'val')
testfolder = os.path.join(outfolder, 'test')
if not os.path.exists(outfolder):
    os.mkdir(outfolder)
if not os.path.exists(trainfolder):
    os.mkdir(trainfolder)
if not os.path.exists(valfolder):
    os.mkdir(valfolder)
if not os.path.exists(testfolder):
    os.mkdir(testfolder)

trainlist = []
vallist = []
testlist = []
    
# Make 2 splits - trainval and test. Split trainval again into train and val
for k,v in rev_map.items():
    split1 = int(np.ceil(len(v) * 0.3))
    np.random.shuffle(v)
    test = v[:split1]
    trainval = v[split1:]
    
    np.random.shuffle(trainval)
    split = int(np.ceil(len(trainval) * 0.3))
    train = trainval[:split]
    val = trainval[split:]
    
    trainlist.extend(train)
    testlist.extend(test)
    vallist.extend(val)
    
    
    for i in train:
        file = os.path.join(infolder, i)
        shutil.copyfile(file, os.path.join(trainfolder, i))
    for i in val:
        file = os.path.join(infolder, i)    
        shutil.copyfile(file, os.path.join(valfolder, i))
    for i in test:
        file = os.path.join(infolder, i)
        shutil.copyfile(file, os.path.join(testfolder, i))
 
print('Total number of Train Images: {}, Val: {}, Test: {}'.format(len(trainlist), len(vallist), len(testlist)))
with open(os.path.join(trainfolder, 'class_mapping.txt'), 'w') as f:
    md = ['{}\t{}'.format(x, file_map[x]) for x in trainlist]
    f.writelines('\n'.join(md))
    
with open(os.path.join(valfolder, 'class_mapping.txt'), 'w') as f:
    md = ['{}\t{}'.format(x, file_map[x]) for x in vallist]
    f.writelines('\n'.join(md))
    
with open(os.path.join(testfolder, 'class_mapping.txt'), 'w') as f:
    md = ['{}\t{}'.format(x, file_map[x]) for x in testlist]
    f.writelines('\n'.join(md))

def assert_correct(folder, orgmap):
    # Assertions
    with open(os.path.join(folder, 'class_mapping.txt')) as f:
        for x,y in [x.split('\t') for x in f.readlines()]:
            assert y.strip() == orgmap[x.strip()]
            
assert_correct(trainfolder, file_map)
assert_correct(valfolder, file_map)
assert_correct(testfolder, file_map)