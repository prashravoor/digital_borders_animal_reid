import paho.mqtt.publish as publish
import cv2
import base64
import time

class MqttClient:
    def __init__(self, channel, hostname):
        self.server = hostname
        self.channel = channel

    def message(self, message):
        publish.single(self.channel, str(message), hostname=self.server)

class MqttWebsocketClient:
    def __init__(self, hostname, port=9001):
        self.server = hostname
        self.port = port

    def message(self, channel, message):
        publish.single(channel, str(message), hostname=self.server, port=self.port, transport='websockets')

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
        self.client.message(detection)


class ImagePublisher:
    def __init__(self, name, server):
        self.server = server
        self.client = MqttClient('{}/image'.format(name), server)
        self.name = name

    def publishImage(self, image):
        self.client.message(cv2.imencode('.jpg', image))

class ImageWsPublisher:
    def __init__(self, name, server, port=9001):
        self.name = name
        self.client = MqttWebsocketClient(server, port)

    def publishImage(self, imageBinData):
        self.client.message('{}/image'.format(self.name), base64.b64encode(imageBinData))
        
class TimedImagePublisher:
    def __init__(self, client, timeout=10):
        self.timeout = 10
        self.client = client
        self.lastSentTime = 0

    def publishImage(self, imageBinData):
        if time.time() - self.lastSentTime > self.timeout:
            self.client.publishImage(imageBinData)
