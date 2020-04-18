import xml.etree.ElementTree as ET
import json
import os
import sys
import cv2
import matplotlib.pyplot as plt

class JaguarAnnotationParser:
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
                + '/../' \
                + os.path.basename(self.filename).split('.')[0] \
                + '.jpg'

    def getWidth(self):
        return int(self.root.find('size')[0].text)

    def getHeight(self):
        return int(self.root.find('size')[1].text)

    def getClassName(self):
        return 'jaguar'

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 2:
        print('Usage: {} <Folder name of annotations>'.format(args[0]))
        exit(1)

    folder = args[1]
    files = ['{}/{}'.format(folder, x.strip()) for x in os.listdir(folder) if x.endswith('.xml')]

    # n = min(10, len(files))
    # files = files[:n]
    n = len(files)

    result = {}
    reid_folder = '{}/../reid'.format(folder)
    for file in files:
        obj = JaguarAnnotationParser(file)
        obj.parse()
        image = cv2.imread(obj.getImageFileName())
        if image is None:
            print('Failed to read image {}: {}'.format(file, obj.getImageFileName()))
            exit(1)
        bboxes = obj.getBoundingBoxes()

        for i in range(len(bboxes)):
            box = bboxes[i]
            subimg = image[box[1]:box[3], box[0]:box[2],] # Index as y,x
            imname = '{}/{}_{}.jpg'.format(reid_folder, os.path.basename(file).split('.')[0], i)
            print(imname)
            cv2.imwrite(imname, subimg)

