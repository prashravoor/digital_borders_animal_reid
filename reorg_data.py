import os
import sys
from shutil import copyfile

args = sys.argv
if not len(args) == 3:
    print('Usage: cmd <Input directory> <output_directory>')
    exit(1)

folder = args[1]
out = args[2]
class_mapping_file = 'normalized_class_mapping.txt'
if not os.path.exists(os.path.join(folder, class_mapping_file)):
    print('Folder doesnt contain metdata file {}'.format(class_mapping_file))
    exit(2)

if os.path.exists(out):
    print('Folder {} already exists'.format(out))
    exit(3)

with open(os.path.join(folder, class_mapping_file)) as f:
    lines = f.readlines()

id_map = { os.path.join(folder, x.split('\t')[0].strip()):x.split('\t')[1].strip() for x in lines}
rev_map = dict()
for k,v in id_map.items():
    if v not in rev_map:
        rev_map[v] = []
    rev_map[v].append(k)

print('Found {} total classes'.format(len(rev_map)))
os.mkdir(out)
# Organizes data by moving each image into a folder of corresponding class name
# Ex:
#  jag
#   0 
#    1.jpg,
#    2.jpg...
#   1
#    3.jpg...
for k,v in rev_map.items():
    path = os.path.join(out,k)
    for val in v:
        if os.path.exists(val):
            if not os.path.exists(path):
                os.mkdir(path)
            copyfile(val, os.path.join(path, os.path.basename(val)))

