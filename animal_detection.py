import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

def non_max_suppression(tf_sess, boxes, probs=None, nms_threshold=0.5, iou_threshold=0.5):
    result = tf.image.non_max_suppression(boxes, probs, max_output_size=5, score_threshold=nms_threshold, iou_threshold=iou_threshold)
    return tf_sess.run(result)

def loadLabels(file):
    with open(file) as f:
        lines = [x.split(',') for x in f.readlines()]
        f.close()

    mapping = {}
    for line in lines:
        mapping[int(line[0])] = line[1].strip()

    return mapping

if __name__ == '__main__':
    input_names = ['image_tensor']
    # IMAGE_PATH = 'amur/plain_reid_train/train/005154.jpg'
    IMAGE_PATH = 'amur_small/003782.jpg'

    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (300, 300))
    trt_graph = get_frozen_graph('ssd/frozen_inference_graph.pb')
    tf_sess = tf.Session()
    tf.import_graph_def(trt_graph, name='')

    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image[None, ...] # Input needs to be 4D, so reshape to add extra dim
    })
    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    labels = loadLabels('ssd/label_mapping.csv')
    # Remove overlapping boxes with non-max suppression, return picked indexes.
    pick = non_max_suppression(tf_sess, boxes, scores)
    
    boxes_pixels = []
    for i in range(len(pick)):
        # scale box to image coordinates
        box = boxes[pick[i]] * np.array([image.shape[0],
                                   image.shape[1], image.shape[0], image.shape[1]])
        box = np.round(box).astype(int)
        boxes_pixels.append(box)
    boxes_pixels = np.array(boxes_pixels)

    for i in pick:
        box = boxes_pixels[i]
        box = np.round(box).astype(int)
        print('Drawing BB: {}'.format(box))
        # Draw bounding box.
        image = cv2.rectangle(
            image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        label = '{}: {:.2f}'.format(labels[int(classes[i])], scores[i])
        # Draw label (class index andbprobability).
        draw_label(image, (box[1], box[0]), label)

    plt.imshow(image[:,:,(2,1,0)])
    plt.show()
