import os
import sys
import numpy as np
import shutil

args = sys.argv
if not len(args) == 2:
    print('Usage: <cmd> <input folder>')
    exit()

infolder = args[1]

images = [x for x in os.listdir(infolder) if x.endswith('.jpg') or x.endswith('.png')]
print('Renaming {} files'.format(len(images)))

counter = 0
int_names = dict()
# Provide integer names for each file
for k in images:
    int_names[k] = '{}.jpg'.format(counter)
    counter += 1

with open(os.path.join(infolder, 'file_map.txt'), 'w') as f:
    f.writelines('\n'.join(['{}\t{}'.format(k,v) for k, v in int_names.items()]))

with open(os.path.join(infolder, 'class_mapping.txt')) as f:
    mapping = {x.split('\t')[0].strip(): x.split('\t')[1].strip() for x in f.readlines()}
  
lines = []
for i in images:
    org = os.path.join(infolder, i)
    mvd = os.path.join(infolder, int_names[i])
    os.rename(org, mvd)
 
# Map new names to old classes
lines = []
for k,v in mapping.items():
    lines.append('{}\t{}'.format(int_names[k], v))

os.rename(os.path.join(infolder, 'class_mapping.txt'), os.path.join(infolder, 'class_mapping_org.txt'))
with open(os.path.join(infolder, 'class_mapping.txt'), 'w') as f:
    f.writelines('\n'.join(lines))