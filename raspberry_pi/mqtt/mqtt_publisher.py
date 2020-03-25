import paho.mqtt.publish as publish
import cv2
import base64
import time
import traceback
import pickle

class MqttClient:
    def __init__(self, channel, hostname):
        self.server = hostname
        self.channel = channel

    def message(self, message):
        publish.single(self.channel, message, hostname=self.server)

class MqttWebsocketClient:
    def __init__(self, hostname, port=9001):
        self.server = hostname
        self.port = port

    def message(self, channel, message):
        publish.single(channel, message, hostname=self.server, port=self.port, transport='websockets')

class TrackPublisher:
    def __init__(self, name, server):
        self.server = server
        self.REGISTRATION = MqttClient('registration', server)
        self.REFRESH = MqttClient('refresh', server)
        self.client = MqttClient(name, server)
        self.name = name

    def register(self):
        self.REGISTRATION.message('{},{}'.format(self.name, self.name))

    def refresh(self):
        self.REFRESH.message('') 

    def publishDetection(self, detection):
        self.client.message(str(detection))


class ImagePublisher:
    def __init__(self, name, server):
        self.server = server
        self.client = MqttClient('{}'.format(name), server)
        self.name = name

    def publishImageWithDetections(self, image, dets):
        try:
            start = time.time()
            message = dict()
            message['detections'] = str(dets)
            message['image'] = cv2.imencode('.jpg', image)[1]
            message = pickle.dumps(message)
            self.client.message(message)
            print('Image transmission time: {:.4f}s'.format(time.time() - start))
        except:
            traceback.print_exc()

    def publishImage(self, image):
        try:
            _,data = cv2.imencode('.jpg', image)
            self.client.message(data.tostring())
        except:
            traceback.print_exc()

class ImageWsPublisher:
    def __init__(self, name, server, port=9001):
        self.name = name
        self.client = MqttWebsocketClient(server, port)

    def publishImage(self, image):
        print('Sending image to channel: {}/image'.format(self.name))
        self.client.message('{}/image'.format(self.name), base64.b64encode(image))

class TimedImagePublisher:
    def __init__(self, client, timeout=10):
        self.timeout = 10
        self.client = client
        self.lastSentTime = 0

    def publishImage(self, imageBinData):
        if time.time() - self.lastSentTime > self.timeout:
            print('Sending Image to channel {}...'.format(self.client.name))
            self.client.publishImage(imageBinData)
            self.lastSentTime = time.time()
