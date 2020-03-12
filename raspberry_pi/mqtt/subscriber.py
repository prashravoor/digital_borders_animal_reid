import paho.mqtt.client as mqtt
import time
from object_tracker import Tracker
from collections import namedtuple, Iterable, defaultdict
from utils import DetectionResult, BoundingBox

TrackedSubject = namedtuple('TrackedSubject', 'tracker timestamp')

MQTT_SERVER = "localhost"
MQTT_REG_PATH = "registration"
MQTT_REFRESH = 'refresh'

class Registration:
    def __init__(self, name, timeout = 10):
        self.name = name
        self.active_tracks = defaultdict(list) # Per class object tracker lists
        self.timeout = timeout # seconds

    def __repr__(self):
        return self.name

    def onEvent(self, client, userdata, msg):
        #print('Got message for device: {}: {}'.format(self.name, msg.payload.decode()))
        detections = eval(msg.payload.decode())

        if not isinstance(detections, Iterable):
            print('Invalid type provided in event, expected iterable, got: {}'.format(type(detections)))
            return

        for det in detections:
            if not isinstance(det, DetectionResult):
                print('Invalid object {}, expected type {}'.format(type(det), DetectionResult))
            else:
                self.addDetection(det)

    def addDetection(self, detection):
        self.active_tracks[detection.classid].append(TrackedSubject(Tracker(detection.bounding_box), time.time()))

    def getActiveTracks(self):
        curtime = time.time()
        tmp = defaultdict(list)
        # Remove inactive tracks
        for k,v in self.active_tracks.items():
            tmp[k] = [x for x in v if (curtime - x.timestamp) < self.timeout]

        self.active_tracks = tmp
        return self.active_tracks.copy()

class CentralServer:
    def __init__(self, server, port=1883):
        self.server = server
        self.port = port
        self.topic_funcs = {MQTT_REG_PATH : self.register, MQTT_REFRESH: self.refresh}
        self.registrations = []

    def register(self, client, userdata, msg):
        parts = msg.payload.decode().split(',')
        if not len(parts) == 2:
            print('Invalid message: {}'.format(msg.payload))
        else:
            print('Registering device {}, Channel: {}'.format(parts[0], parts[1]))
            if parts[1] in self.topic_funcs:
                print('Warning: Device already registered...')
            else:
                client.subscribe(parts[1])
                reg = Registration(parts[0])
                self.registrations.append(reg)
                self.topic_funcs[parts[1]] = reg.onEvent

    def refresh(self, client, userdata, msg):
        print()
        print('-----------------')
        print('Current active devices: {}'.format(self.registrations))
        for reg in self.registrations:
            tracks = reg.getActiveTracks()
            print('Total {} active tracks found for device: {}'.format(len(tracks), reg))
            for t,v in tracks.items():
                print('Found Total entities with class id {}: {}'.format(t, len(v)))
        print('-----------------')
        print()

    def on_message(self, client, userdata, msg):
        try:
            if msg.topic in self.topic_funcs:
                self.topic_funcs[msg.topic](client, userdata, msg)
            else:
                print('Not subscribed to topic {}, ignoring...'.format(msg.topic))
        except BaseException as e:
            print('Failed..: {}'.format(e))

    def onConnect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        client.subscribe(MQTT_REG_PATH)
        client.subscribe(MQTT_REFRESH)

    def run(self):
        client = mqtt.Client()
        client.on_connect = self.onConnect
        #client.on_message = self.onRegistration
        client.on_message = self.on_message

        client.connect(self.server, self.port, 60)
        client.loop_forever()

if __name__ == '__main__':
    BaseException
    server = CentralServer(MQTT_SERVER)
    server.run()
