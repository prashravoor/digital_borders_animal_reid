import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

def drawLabel(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)
    return image

def drawBoundingBoxWithLabel(image, res, labels, color=(0,255,0), thickness=2):
    top_left = (res.bounding_box.xmin, res.bounding_box.ymin)
    bottom_right = (res.bounding_box.xmax, res.bounding_box.ymax)
    label = '{}: {:.2f}'.format(labels[res.classid], res.confidence)
    return drawBoundingBoxLabel(image, label, res, color, thickness)

def drawBoundingBoxLabel(image, label, res, color=(0,255,0), thickness=2):
    top_left = (res.bounding_box.xmin, res.bounding_box.ymin)
    bottom_right = (res.bounding_box.xmax, res.bounding_box.ymax)
    image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    if top_left[0] < 20:
        top_left = (top_left[0]+20, top_left[1])
    if top_left[1] < 20:
        top_left = (top_left[0], top_left[1]+20)

    return drawLabel(image, top_left, label, thickness=thickness)

def iou(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def loadImageAsTensor(imageFile):
    image = tf.io.read_file(imageFile)
    return tf.image.decode_jpeg(image)
