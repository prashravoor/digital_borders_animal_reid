from object_detection import ObjectDetector
import sys
import picamera
import time
import picamera.array

if __name__ == '__main__':
    args = sys.argv

    if not len(args) == 2:
        print('Model path needed')
        exit()

    modelpath = args[1]
    detector = ObjectDetector(modelpath, numthreads=4)
    detector.loadModel()


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

