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

# Remove identities with less than 2 images
rev_map = {k:v for k,v in rev_map.items() if len(v) > 1}
print('Total number of identities: {}'.format(len(rev_map)))

trainfolder = os.path.join(outfolder, 'train')
testfolder = os.path.join(outfolder, 'test')
if not os.path.exists(outfolder):
    os.mkdir(outfolder)
if not os.path.exists(trainfolder):
    os.mkdir(trainfolder)
if not os.path.exists(testfolder):
    os.mkdir(testfolder)

trainlist = []
testlist = []
    
# Use 20% of the identities as test set
num_ids = len(rev_map)
split = int(np.ceil(num_ids * 0.2))
test_ids = np.random.choice(list(rev_map.keys()), split, replace=False)
train_ids = [x for x in rev_map.keys() if not x in test_ids]

print('Number of train ids: {}, Test Ids: {}'.format(len(train_ids), len(test_ids)))
for id in train_ids:
    trainlist.extend(rev_map[id])
for id in test_ids:
    testlist.extend(rev_map[id])
    
for i in trainlist:
    file = os.path.join(infolder, i)
    shutil.copyfile(file, os.path.join(trainfolder, i))

for i in testlist:
    file = os.path.join(infolder, i)
    shutil.copyfile(file, os.path.join(testfolder, i))

print('Total number of Train Images: {}, Test: {}'.format(len(trainlist), len(testlist)))
with open(os.path.join(trainfolder, 'class_mapping.txt'), 'w') as f:
    md = ['{}\t{}'.format(x, file_map[x]) for x in trainlist]
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
assert_correct(testfolder, file_map)