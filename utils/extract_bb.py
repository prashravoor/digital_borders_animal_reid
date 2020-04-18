import cv2
import cv2.dnn as dnn
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import time
import argparse
import os


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

classes = None
COLORS = None

if '__main__' == __name__:
    if len(sys.argv) < 2:
        print('Usage: {}, [{}], [{}], [{}]'.format('Image File', 'Network Config File', 'Network Weights File', 'Classes File'))
        exit(1)

# handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
            help = 'path to input image')
    ap.add_argument('-c', '--config', required=False,
            help = 'path to yolo config file', default=os.path.abspath('../darknet/darknet/cfg/yolov3.cfg'))
    ap.add_argument('-w', '--weights', required=False,
            help = 'path to yolo pre-trained weights', default=os.path.abspath('../darknet/darknet/yolov3.weights'))
    ap.add_argument('-cl', '--classes', required=False,
            help = 'path to text file containing class names', default=os.path.abspath('../darknet/darknet/data/coco.names'))
    args = ap.parse_args()
    
    imfile = args.image
    # imfile = sys.argv[1]
    # cfg = sys.argv[2]
    # weights = sys.argv[3]
    # classfile = sys.argv[4]
    cfg = args.config
    weights = args.weights
    classfile = args.classes

    try:
        with open(classfile) as f:
            classes = [line.strip() for line in f.readlines()]
            f.close()
    except IOError as e:
        print(e)
        exit(1)

    COLORS = np.random.uniform(0,255,size=(len(classes), 3))
    image = cv2.imread(imfile)
    scale = 1/255.0
    width = image.shape[1]
    height = image.shape[0]
    conf_threshold = 0.75
    nms_threshold = 0.45
    net_width = 608
    net_height = 608

    try:
        net = dnn.readNet(cfg, weights)
    except IOError as e:
        print('Invalid Configuration or Weights File: {}'.format(e))
        exit(1)

    start = time.time()
    blob = dnn.blobFromImage(image, scale, (net_width,net_height), (0,0,0), swapRB=False, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    end = time.time()
    print('Time taken for prediction: {}'.format(end - start))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = (center_x - w/2.0)#  * width
                y = (center_y - h/2.0)#  * height
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    test = copy.copy(image)
    bboxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = box[2]
        h = box[3]

        bboxes.append((image[y:y+h, x:w+x], class_ids[i]))
        draw_bounding_box(test, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    plt.imshow(test)
    if len(bboxes) > 0:
        fig,axes = plt.subplots(1, len(bboxes))
        for i in range(len(bboxes)): 
            if len(bboxes) == 1:
                ax = axes
            else:
                ax = axes[i]

            ax.imshow(bboxes[i][0])
            ax.set_title(classes[bboxes[i][1]])

    plt.show()



