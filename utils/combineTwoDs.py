import sys
import numpy as np
import os
from shutil import copyfile

args = sys.argv
if len(args) < 3:
    print('Usage: cmd <class mapping file 1> <class mapping file 2> <output folder>')
    exit(1)

file1 = args[1]
file2 = args[2]
outfolder = args[3]

with open(file1) as f:
    mapping1 = {x.split('\t')[0].strip(): int(x.split('\t')[1].strip()) for x in f.readlines()}

with open(file2) as f:
    mapping2 = {x.split('\t')[0].strip(): int(x.split('\t')[1].strip()) for x in f.readlines()}

if not os.path.exists(outfolder):
    os.mkdir(outfolder)
    
newmapping = dict()
outmappingfile = os.path.join(outfolder, 'class_mapping.txt')

totalsize = len(mapping1) + len(mapping2)
print('Total files in File 1: {}, File 2: {}, Combined: {}'.format(len(mapping1), len(mapping2), totalsize))

for k,v in mapping1.items():
    basename = os.path.basename(k)
    newname = os.path.join(outfolder, basename)
    
    copyfile(k, newname)
    newmapping[basename] = v
    
assert len(newmapping) == len(mapping1)

counter = 0
for k,v in mapping2.items():
    basename = os.path.basename(k)
    newname = os.path.join(outfolder, basename)
    
    if os.path.exists(newname):
        #print('File {} already exists!! Renaming'.format(basename))
        basename = 'cstar_{}'.format(basename)
        newname = os.path.join(outfolder, basename)
        
    copyfile(k, newname)
    newmapping[basename] = v

with open(outmappingfile, 'w') as f:
    f.writelines('\n'.join(['{}\t{}'.format(k,v) for k,v in newmapping.items()]))

assert totalsize == len(newmapping), 'Totalsize: {}, actual size: {}'.format(totalsize, len(newmapping))