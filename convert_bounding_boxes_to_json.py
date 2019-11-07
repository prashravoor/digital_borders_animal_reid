import xml.etree.ElementTree as ET
import json
import os
import sys

args = sys.argv
if not len(args) == 2:
    print('Usage: {} <Folder name of annotations>'.format(args[0]))
    exit(1)

folder = args[1]
files = ['{}/{}'.format(folder, x.strip()) for x in os.listdir(folder)]

# n = min(10, len(files))
# files = files[:n]
n = len(files)

result = {}
for file in files:
    root = ET.parse(file).getroot()
    file_name = os.path.basename(file)
    image_name = file_name.split('.')[0]

    bboxes = []
    for bbox in root.iter('bndbox'):
        box = []
        for point in bbox:
            box.append(int(point.text))
        bboxes.append(box)

    result[image_name] = bboxes

with open('{}.json'.format(folder), 'w') as f:
    json.dump(result, f)
    f.close()
