from object_detection import ObjectDetector
import sys
import picamera
import time
import picamera.array
from mqtt_publisher import TrackPublisher
import cv2
import threading

DEVICE_NAME = 'raspberrypi1'
LABELS = {0: 'Unknown', 1: 'Tiger', 2: 'Elephant', 3: 'Jaguar', 4: 'Human'}

def drawBbOnImage(image, detections):
# Get bounding box coordinates and draw box
# Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
    for det in detections:
        ymin = det.bounding_box.ymin
        xmin = det.bounding_box.xmin 
        ymax = det.bounding_box.ymax 
        xmax = det.bounding_box.xmax 

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

# Draw label
        if int(det.classid) < len(LABELS):
            object_name = LABELS[int(det.classid)]
        else:
            object_name = 'Unknown'

        label = '%s: %d%%' % (object_name, int(det.confidence*100))
        labelSize, baseLine = cv2.getTextSize(label, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) 
        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10),
                     (xmin+labelSize[0], label_ymin+baseLine-10),
                     (255, 255, 255),
                     cv2.FILLED) # Draw white box to put label text in
        cv2.putText(image, label, 
                (xmin, label_ymin-7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0), 2) # Draw label text

def sendImageAsync(client, image):
    threading.Thread(target=client.publishImage, args=(image)).start()

if __name__ == '__main__':
    args = sys.argv

    if len(args) < 3:
        print('Usage: cmd <Model path> <MQTT server IP> [display]')
        exit()

    modelpath = args[1]
    server = args[2]
    detector = ObjectDetector(modelpath, numthreads=4)
    detector.loadModel()

    display = False
    if len(args) > 3 and args[3] == 'display':
        display = True

    publisher = TrackPublisher(DEVICE_NAME, server)
    publisher.register()
    imagePublisher = ImagePublisher(DEVICE_NAME, server)

    print('Registered with MQTT server, publishing messages on channel: {}'.format(DEVICE_NAME))

    with picamera.PiCamera(resolution=(640,480), framerate=30) as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.start_preview()
            time.sleep(2) # Camera warmup
            for _ in camera.capture_continuous(stream, format='rgb'):
                image = stream.array
                image = image[:,:,(2,1,0)]
                stream.truncate()
                stream.seek(0)
                start = time.time()
                results = detector.getBoundingBoxes(image)
                if len(results) > 0:
                        publisher.publishDetection(results)
                if display:
                    print('Got results: {}, Time: {:.4f}s'.format(results, time.time() - start))
                    image = cv2.resize(image, 
                            (detector.IMG_HEIGHT, detector.IMG_WIDTH))
                    if len(results) > 0:
                        drawBbOnImage(image, results)
                        sendImageAsync(imagePublisher, image)

                    cv2.imshow('test', image)
                    cv2.waitKey(1)

