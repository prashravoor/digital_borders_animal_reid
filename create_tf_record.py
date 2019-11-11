import tensorflow as tf
from collections import namedtuple
import cv2
from convert_bounding_boxes_to_json import AmurAnnotationParser 
import sys
import os

TfRecordFormat = namedtuple('TfRecordFormat', 'file width height detections')
Detection = namedtuple('BoundingBox', 'xmin ymin xmax ymax classid classname')

def int64_feature(value):
    return int64_list_feature([value])

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return bytes_list_feature([value])

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_record(tfrecord):
    height = tfrecord.height # Image height
    width = tfrecord.width # Image width
    filename = tfrecord.file # Filename of the image. Empty if image is not from file

    encoded_image_data = tf.io.read_file(tfrecord.file) # Encoded image bytes
    image_format = b'jpeg'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    for box in tfrecord.detections:
        xmins.append(box.xmin / float(width))
        xmaxs.append(box.xmax / float(width))
        ymins.append(box.ymin / float(height))
        ymaxs.append(box.ymax / float(height))
        classes_text.append(box.classname)
        classes.append(box.classid)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(bytes(filename, 'utf-8')),
      'image/source_id': bytes_feature(bytes(filename, 'utf-8')),
      'image/encoded': bytes_feature(encoded_image_data.numpy()),
      'image/format': bytes_feature(image_format),
      'image/object/bbox/xmin': float_list_feature(xmins),
      'image/object/bbox/xmax': float_list_feature(xmaxs),
      'image/object/bbox/ymin': float_list_feature(ymins),
      'image/object/bbox/ymax': float_list_feature(ymaxs),
      'image/object/class/text': bytes_list_feature(classes_text),
      'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example

args = sys.argv
if not len(args) == 2:
    print('Usage: {} <Folder containing annotations>'.format(args[0]))
    exit(1)
folder = args[1]

files = ['{}/{}'.format(folder, x) for x in os.listdir(folder) if x.endswith('.xml')]
n = min(5, len(files))
files = files[:n]

TIGER_CLASS_ID = 1
writer = tf.io.TFRecordWriter('tfrfile')
for file in files:
    parser = AmurAnnotationParser(file)
    parser.parse()
    boxes = []
    for box in parser.getBoundingBoxes():
        boxes.append(Detection(*box, TIGER_CLASS_ID, bytes(parser.getClassName(), 'utf-8')))
    
    tf_example = create_tf_record(TfRecordFormat(parser.getImageFileName(), parser.getWidth(), parser.getHeight(), boxes))
    writer.write(tf_example.SerializeToString())

writer.close()


