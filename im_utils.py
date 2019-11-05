import matplotlib.pyplot as plt
import cv2

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
    image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return drawLabel(image, top_left, label)
