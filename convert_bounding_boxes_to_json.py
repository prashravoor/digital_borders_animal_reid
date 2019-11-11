import xml.etree.ElementTree as ET
import json
import os
import sys

class AmurAnnotationParser:
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        self.root = ET.parse(self.filename).getroot()

    def getBoundingBoxes(self):
        bboxes = []
        for bbox in self.root.iter('bndbox'):
            box = []
            for point in bbox:
                box.append(int(point.text))
            bboxes.append(box)
        return bboxes

    def getImageFileName(self):
        return os.path.dirname(self.filename) \
                + '/../trainval/' \
                + os.path.basename(self.filename).split('.')[0] \
                + '.jpg'

    def getWidth(self):
        return int(self.root.find('size')[0].text)

    def getHeight(self):
        return int(self.root.find('size')[1].text)

    def getClassName(self):
        return 'Tiger'

if __name__ == '__main__':
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
        obj = AmurAnnotationParser(file)
        obj.parse()
        image_name = os.path.basename(file).split('.')[0]
        result[image_name] = obj.getBoundingBoxes()

    with open('{}.json'.format(folder), 'w') as f:
        json.dump(result, f)
        f.close()
