import sys
import os
import shutil
import numpy as np

args = sys.argv
if not len(args) == 3:
    print('Usage: <cmd> <DS folder> <new_folder>')
    exit()

np.random.seed(2149)

infolder = args[1]
outfolder = args[2]

labels = 'normalized_class_mapping.txt'
if not os.path.exists(os.path.join(infolder, labels)):
    labels = 'class_mapping.txt'


images = [x for x in os.listdir(infolder) if x.endswith('.jpg')]
with open(os.path.join(infolder, labels)) as f:
    file_map = {x.split('\t')[0].strip() : x.split('\t')[1].strip() for x in f.readlines()}

file_map = {k:v for k,v in file_map.items() if k in images}

print('Total number of valid images: {}'.format(len(file_map)))

rev_map = dict()
for k,v in file_map.items():
    if not v in rev_map:
        rev_map[v] = []
    rev_map[v].append(k)

print('Total number of individuals in dataset: {}'.format(len(rev_map)))

min_samples = 3
max_samples = 20

rev_map = {k:v for k,v in rev_map.items() if len(v) >= min_samples}
print('Individuals after removing those which dont have {} samples: {}'.format(min_samples, len(rev_map)))

num_images = 0
new_map = dict()
for k,v in rev_map.items():
# Randomly select subset of images
    new_set = np.random.choice(v, min(len(v), max_samples), replace=False)
    new_map[k] = new_set
    num_images += len(new_set)

print('Total images after adjusting dataset: {}'.format(num_images))

print('Copying files...')
if not os.path.exists(outfolder):
    os.mkdir(outfolder)

lines = []
for k,v in new_map.items():
    lines.extend(['{}\t{}'.format(x,k) for x in v])
    for x in v:
        shutil.copyfile(os.path.join(infolder, x), os.path.join(outfolder, x))

with open(os.path.join(outfolder, labels), 'w') as f:
    f.writelines('\n'.join(lines))
