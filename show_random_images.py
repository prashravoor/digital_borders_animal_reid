import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.gridspec as gridspec


NUM_IMAGES = 16
amur = 'amur/plain_reid_train/train'
elp = 'ELPephants/images'
amur_det = 'amur/detection_train/trainval'
jaguar = 'jaguars/reid'
jaguar_det = 'jaguars/'
new_size = (416,416)

def loadLabels(folder):
    result = {}
    if 'amur' in folder:
        file = '{}/{}'.format(folder, 'reid_list_train.csv')
        if not os.path.exists(file):
            return result
        with open(file) as f:
            lines = [x.strip() for x in f.readlines()]
            f.close()

        for line in lines:
            parts = line.split(',')
            result['{}/{}'.format(folder, parts[1])] = parts[0]
    else:
        file = '{}/{}'.format(folder, '../class_mapping.txt')
        names = [x for x in os.listdir(elp) if x.endswith('.jpg')]
        temp = {}
        with open(file) as f:
            lines = [x.strip() for x in f.readlines()]
            f.close()
        for line in lines:
            parts = line.split('\t')
            temp[parts[0]] = parts[1]
        
        for name in names:
            result['{}/{}'.format(folder, name)] = temp[name.split('_')[0]]
    return result 

def showImagesInFolder(folder, numImages):
    images = ['{}/{}'.format(folder,x) for x in os.listdir(folder) if x.endswith('.jpg')]
    rows = int(numImages ** 0.5)
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(rows, rows)
    gs.update(wspace=0, hspace=0.25)

    labels = loadLabels(folder)
    to_show = np.random.choice(range(len(images)), numImages)
    for i in range(numImages):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        imname = images[to_show[i]]
        image = cv2.imread(imname)
        image = cv2.resize(image, new_size)
        image = image[:,:,(2,1,0)] # CV2 reads image in BGR. Convert to RGB
        #if not imname in labels:
        #    label = 'Test'
        #else:
        #    label = labels[imname]
        #ax.set_title('Id: {}'.format(label))
        ax.imshow(image)
    gs.tight_layout(fig)

def showSubjectImages(folder, numImages):
    labels = loadLabels(folder)
    new_size = (416,416)

    mapping = {}
    # Construct reverse map
    for key,value in labels.items():
        if not value in mapping:
            mapping[value] = []
        mapping[value].append(key)

    subject = np.random.choice(np.array([x for x in mapping.keys()]).astype(int))
    sub_images = mapping[str(subject)]

    rows = int(numImages ** 0.5)
    numImages = min(numImages, len(sub_images))

    fig = plt.figure(figsize=(10,10), num='Id: {}'.format(subject))
    gs = gridspec.GridSpec(rows, rows)

    k = 0
    for i in np.random.choice(range(len(sub_images)), numImages):
        ax = plt.subplot(gs[k])
        k += 1
        imname = sub_images[i]
        image = cv2.imread(imname)
        if image is None:
            continue
        image = cv2.resize(image, new_size)
        image = image[:,:,(2,1,0)] # CV2 reads image in BGR. Convert to RGB
        ax.imshow(image)
        plt.axis('off')
    gs.tight_layout(fig)


if __name__ == '__main__':
    showImagesInFolder(elp, NUM_IMAGES)
    showImagesInFolder(amur, NUM_IMAGES)
    showImagesInFolder(amur_det, NUM_IMAGES)
    showImagesInFolder(jaguar, NUM_IMAGES)
    showImagesInFolder(jaguar_det, NUM_IMAGES)

    SUB_IMAGES = 9
    showSubjectImages(amur, SUB_IMAGES)
    showSubjectImages(elp, SUB_IMAGES)
    showSubjectImages(jaguar, SUB_IMAGES)
    plt.show()
