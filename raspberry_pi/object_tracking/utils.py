from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', 'ymin xmin ymax xmax')
DetectionResult = namedtuple('DetectionResult', 'bounding_box confidence classid')
