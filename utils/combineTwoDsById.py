import sys
import numpy as np
import os
from shutil import copyfile

np.random.seed(42)

args = sys.argv
if len(args) < 5:
    print('Usage: cmd <class mapping train file> <class mapping test file> <train out> <test out>')
    exit(1)

file1 = args[1]
file2 = args[2]
trainout = args[3]
testout = args[4]

with open(file1) as f:
    mapping1 = {os.path.join(os.path.dirname(file1), x.split('\t')[0].strip()): int(x.split('\t')[1].strip()) for x in f.readlines()}
    
with open(file2) as f:
    mapping2 = {os.path.join(os.path.dirname(file2), x.split('\t')[0].strip()): int(x.split('\t')[1].strip()) for x in f.readlines()}

totalsize = len(mapping1) + len(mapping2)
print('Total files in File 1: {}, File 2: {}, Combined: {}'.format(len(mapping1), len(mapping2), totalsize))

revmap = dict()
for k,v in mapping1.items():
    if not v in revmap:
        revmap[v] = []
    revmap[v].append(k)
    
for k,v in mapping2.items():
    if not v in revmap:
        revmap[v] = []
    revmap[v].append(k)

revmap_tot_images = sum([len(v) for _,v in revmap.items()])
print ('Total identities: {}, Total images: {}'.format(len(revmap), revmap_tot_images))
assert revmap_tot_images == totalsize, 'Images in revmap {} do not match total images: {}'.format(revmap_tot_images, totalsize)

ids = list(revmap.keys())

num_test = int(np.ceil(0.2 * len(ids))) # 20% ids as test
test_ids = np.random.choice(ids, num_test, replace=False)
train_ids = [x for x in ids if not x in test_ids]

print('Train Ids: {}, Test Ids: {}'.format(len(train_ids), len(test_ids)))

test_map = {k:v for k,v in revmap.items() if k in test_ids}
train_map = {k:v for k,v in revmap.items() if k in train_ids}

if not os.path.exists(trainout):
    os.mkdir(trainout)
    
if not os.path.exists(testout):
    os.mkdir(testout)

trainout_mappingfile = os.path.join(trainout, 'class_mapping.txt')
testout_mappingfile = os.path.join(testout, 'class_mapping.txt')

trainout_newmapping = dict()
for k,v in train_map.items():
    for f in v:
        basename = os.path.basename(f)
        newname = os.path.join(trainout, basename)
        
        if os.path.exists(newname):
            #print('File {} already exists!! Renaming'.format(basename))
            basename = 'cstar_{}'.format(basename)
            newname = os.path.join(trainout, basename)
            
        copyfile(f, newname)
        trainout_newmapping[basename] = k

with open(trainout_mappingfile, 'w') as f:
    f.writelines('\n'.join(['{}\t{}'.format(k,v) for k,v in trainout_newmapping.items()]))

train_imgs = len(trainout_newmapping)    
    
testout_newmapping = dict()
for k,v in test_map.items():
    for f in v:
        basename = os.path.basename(f)
        newname = os.path.join(testout, basename)
        
        if os.path.exists(newname):
            #print('File {} already exists!! Renaming'.format(basename))
            basename = 'cstar_{}'.format(basename)
            newname = os.path.join(testout, basename)
            
        copyfile(f, newname)
        testout_newmapping[basename] = k

with open(testout_mappingfile, 'w') as f:
    f.writelines('\n'.join(['{}\t{}'.format(k,v) for k,v in testout_newmapping.items()]))
    
test_imgs = len(testout_newmapping)
assert test_imgs + train_imgs == totalsize, 'The final list of images {} + {} = {} != total images {}'.format(train_imgs, test_imgs, train_imgs + test_imgs, totalsize)