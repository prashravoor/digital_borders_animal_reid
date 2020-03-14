from utils import BoundingBox, DetectionResult
from mqtt_publisher import TrackPublisher
import sys
import time

def test_left_then_right(client):
    prev = BoundingBox(0,50,12,62)
    detection = DetectionResult(prev, 0.9, 1)

    for _ in range(3):
        client.publishDetection([detection])
        prev = BoundingBox(prev.ymin, prev.xmin-3, prev.ymax, prev.xmax-3)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        time.sleep(1)

    for _ in range(3):
        client.publishDetection([detection])
        prev = BoundingBox(prev.ymin, prev.xmin+3, prev.ymax, prev.xmax+3)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        time.sleep(1)

def test_multi_device_right_right(client):
    org = BoundingBox(0,50,12,62)
    prev = org 
    detection = DetectionResult(prev, 0.9, 1)

    # Device one
    for _ in range(3):
        client.publishDetection([detection])
        prev = BoundingBox(prev.ymin, prev.xmin+3, prev.ymax, prev.xmax+3)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        time.sleep(1)

    # Device 2
    orgname = client.client.channel
    client.client.channel = 'tester2'
    client.register()

    for _ in range(3):
        client.publishDetection([detection])
        prev = BoundingBox(prev.ymin, prev.xmin+3, prev.ymax, prev.xmax+3)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        time.sleep(1)

    client.client.channel = orgname

def test_multi_device_incoming(client):
    org = BoundingBox(20,50,92,122)
    prev = org 
    detection = DetectionResult(prev, 0.9, 1)

    # Device one
    for _ in range(3):
        client.publishDetection([detection])
        prev = BoundingBox(prev.ymin+3, prev.xmin-3, prev.ymax+3, prev.xmax-3)
        prev = BoundingBox(prev.ymin-1, prev.xmin-1, prev.ymax+1, prev.xmax+1)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        time.sleep(1)

    # Device 2
    orgname = client.client.channel
    client.client.channel = 'tester2'
    client.register()

    prev = org 
    for _ in range(3):
        client.publishDetection([detection])
        prev = BoundingBox(prev.ymin-1, prev.xmin-1, prev.ymax+1, prev.xmax+1)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        detection = DetectionResult(prev, detection.confidence, detection.classid)
        time.sleep(1)

    client.client.channel = orgname



if __name__ == '__main__':
    server = 'localhost'

    client = TrackPublisher('tester', server)
    client.register()

    funcs = [#test_left_then_right,
             #test_multi_device_right_right,
             test_multi_device_incoming,
            ] 

    print('Running total {} tests..'.format(len(funcs)))
    failed_funcs = []
    for f in funcs:
        try:
            f(client)
            client.refresh()
        except AssertionError as e:
            print('Test: {} failed: {}'.format(f.__name__, e))
            failed_funcs.append(f.__name__)

    print('------- Summary --------')
    print('Total Tests: {}, Passed: {}, Failed: {}'.format(len(funcs), len(funcs) - len(failed_funcs), len(failed_funcs)))
    if len(failed_funcs) > 0:
        print('Failed: {}'.format(failed_funcs))
