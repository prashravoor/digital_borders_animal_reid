from object_detection import ObjectDetector
import sys
import picamera
import time
import picamera.array
from mqtt_client import TrackPublisher

DEVICE_NAME = 'raspberrypi1'

if __name__ == '__main__':
    args = sys.argv

    if not len(args) == 3:
        print('Usage: cmd <Model path> <MQTT server IP>')
        exit()

    modelpath = args[1]
    server = args[2]
    detector = ObjectDetector(modelpath, numthreads=4)
    detector.loadModel()

    publisher = TrackPublisher(DEVICE_NAME, server)
    publisher.register()
    print('Registered with MQTT server, publishing messages on channel: {}'.format(DEVICE_NAME))

    with picamera.PiCamera(resolution=(640,480), framerate=30) as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.start_preview()
            time.sleep(2) # Camera warmup
            for _ in camera.capture_continuous(stream, format='rgb'):
                image = stream.array
                stream.truncate()
                stream.seek(0)
                start = time.time()
                results = detector.getBoundingBoxes(image)
                print('Got results: {}, Time: {:.4f}s'.format(results, time.time() - start))
                for result in results:
                    publisher.publishDetection(result)

