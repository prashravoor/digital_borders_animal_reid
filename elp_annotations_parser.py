import xml.etree.ElementTree as ET
import json
import os
import sys
import cv2
import matplotlib.pyplot as plt

class ElephantAnnotationParser:
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
                + 'images/' \
                + os.path.basename(self.filename).split('.')[0] \
                + '.jpg'

    def getWidth(self):
        return int(self.root.find('size')[0].text)

    def getHeight(self):
        return int(self.root.find('size')[1].text)

    def getClassName(self):
        return 'jaguar'

def select_largest_bbox(res):
    # Hack - this is actually supposed to calcualte the area of each box and return the largest one
    # right now for all images, the first is the largest. :)
    return res[0]

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 3:
        print('Usage: {} <Folder name of annotations> <out folder>'.format(args[0]))
        exit(1)

    folder = args[1]
    outfolder = args[2]
    files = ['{}/{}'.format(folder, x.strip()) for x in os.listdir(folder) if x.endswith('.xml')]

    #n = min(100, len(files))
    #files = files[:n]
    n = len(files)

    result = {}
    reid_folder = outfolder
    if not os.path.exists(reid_folder):
        os.mkdir(reid_folder)

    for file in files:
        obj = ElephantAnnotationParser(file)
        obj.parse()
        bboxes = obj.getBoundingBoxes()
        image = cv2.imread(obj.getImageFileName())
        if image is None:
            print('Failed to read image {}: {}'.format(file, obj.getImageFileName()))
            exit(1)
        if len(bboxes) > 1:
                box = select_largest_bbox(bboxes)
        else:
            box = bboxes[0]

        subimg = image[box[1]:box[3], box[0]:box[2],] # Index as y,x
        imname = '{}.jpg'.format(os.path.join(reid_folder, os.path.basename(file).split('.')[0]))
        print(imname)
        cv2.imwrite(imname, subimg, [cv2.IMWRITE_JPEG_QUALITY, 50])
