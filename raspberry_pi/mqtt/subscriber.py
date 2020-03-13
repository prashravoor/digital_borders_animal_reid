import paho.mqtt.client as mqtt
import time
from object_tracker import Tracker, Tracklet, mergeTracklets, getDirections
from collections import namedtuple, Iterable, defaultdict
from utils import DetectionResult, BoundingBox

TrackedSubject = namedtuple('TrackedSubject', 'tracker timestamp')

MQTT_SERVER = "localhost"
MQTT_REG_PATH = "registration"
MQTT_REFRESH = 'refresh'
LABELS = {1: 'Tiger', 2: 'Elephant', 3: 'Jaguar', 4: 'Human'}

class SubjectMonitor:
    def __init__(self, timeout=3000):
        self.detections = dict()
        self.timeout = timeout

    def _getIdentifier(self, classid):
        # Right now, use class id for mapping tracklets. Replace wth actual identification code here
        name = '{}_{}'.format(LABELS[classid], 0)
        return name

    def addDetection(self, detection, idf=None):
        curtime = time.time()
        if idf is None:
            idf = self._getIdentifier(detection.classid)

        if not idf in self.detections:
            print('Starting new track for id: {}'.format(idf))
            self.detections[idf] = TrackedSubject(Tracker(detection.bounding_box), curtime)
        else:
            print('Found existing id as: {}'.format(idf))
            tmp = self.detections[idf]
            tmp.tracker.addFrame(detection.bounding_box)
            self.detections[idf] = TrackedSubject(tmp.tracker, curtime)

        return idf

    def getActiveDetections(self):
        # Remove stale tracklets and return all active detections
        curtime = time.time()
        # Remove inactive tracks
        self.detections = {k:v for k,v in self.detections.items() if (curtime - v.timestamp) < self.timeout}

        return {k:v.tracker.getTracklets() for k,v in self.detections.items()} 

class CrossCameraMonitor:
    def __init__(self, timeout=3000):
        self.registered = dict()
        self.timeout = timeout

    def _getIdentifier(self, detection):
        # Right now, use class id for mapping tracklets. Replace wth actual identification code here
        name = '{}_{}'.format(LABELS[detection.classid], 0)
        return name

    def addRegistration(self, name):
        if name in self.registered:
            print('{} already registered, ignoring...')
        else:
            self.registered[name] = SubjectMonitor(self.timeout)

    def addDetection(self, name, detection):
        idf = self._getIdentifier(detection)
        print('Adding detection for {} as {}'.format(name, idf))
        if name in self.registered:
            self.registered[name].addDetection(detection, idf)
        else:
            print('No registered device for {}'.format(name))

    def getActiveDetections(self):
        # Combine tracklets from all registered device, sort them by time first
        results = defaultdict(list)
        for name,reg in self.registered.items():
            dets = reg.getActiveDetections()
            for k,v in dets.items():
                # Index the detections by classid, across all cameras
                results[k].append((name, v))

        return results

class DeviceMonitor:
    def __init__(self, name, centralMonitor, timeout = 10):
        self.name = name
        self.active_tracks = dict() # Per class object tracker lists
        self.timeout = timeout # seconds
        self.centralMonitor = centralMonitor
        self.monitor = SubjectMonitor(timeout)

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
        self.monitor.addDetection(detection)
        self.centralMonitor.addDetection(self.name, detection)

    def getActiveTracks(self):
        return self.monitor.getActiveDetections()
    
class CentralServer:
    def __init__(self, server, port=1883):
        self.server = server
        self.port = port
        self.topic_funcs = {MQTT_REG_PATH : self.register, MQTT_REFRESH: self.refresh}
        self.registrations = []
        self.monitor = CrossCameraMonitor()

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
                reg = DeviceMonitor(parts[0], self.monitor)
                self.registrations.append(reg)
                self.topic_funcs[parts[1]] = reg.onEvent
                self.monitor.addRegistration(parts[0])

    def refresh(self, client, userdata, msg):
        print()
        print('-----------------')
        print('Current active devices: {}'.format(self.registrations))
        for reg in self.registrations:
            tracks = reg.getActiveTracks()
            print('Total {} active tracks found for device: {}'.format(len(tracks), reg))
            for t,v in tracks.items():
                # print('Found Total entities with class id {}: {}'.format(t, len(v)))
                print('Movements of {}: {}'.format(t, getDirections(v)))

        print()
        print('History of tracks:')
        for idf,tracks in self.monitor.getActiveDetections().items():
            print('Identity: {}, Movement History: {}'.format(idf, tracks))

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
    server = CentralServer(MQTT_SERVER)
    server.run()
