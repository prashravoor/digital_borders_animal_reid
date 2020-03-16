import paho.mqtt.client as mqtt
import time
from object_tracker import Tracker, Tracklet, mergeTracklets, getDirections
from collections import namedtuple, Iterable, defaultdict
from utils import DetectionResult, BoundingBox
from mqtt_subscriber import MqttSubscriber
from mqtt_publisher import MqttWebsocketClient,ImageWsPublisher,TimedImagePublisher
import threading
import json
from im_utils import iou
import numpy as np
import traceback

TrackedSubject = namedtuple('TrackedSubject', 'tracker timestamp')
DetectionInfo = namedtuple('DetectionInfo', 'name activeDetections deterState')
DetectionHistory = namedtuple('DetectionHistory', 'name lastSeen travelDirection lastSeenTime')
CrossCamDetection = namedtuple('CrossCamDetection', 'name lastSeen timestamp bbox')

MQTT_SERVER = "localhost"
MQTT_REG_PATH = "registration"
MQTT_REFRESH = 'refresh'
MQTT_FRONTEND_BASE = 'frontend'
MQTT_FRONTEND_REG = '{}/registration'.format(MQTT_FRONTEND_BASE)
MQTT_FRONTEND_DET_HISTORY = '{}/detectionHistory'.format(MQTT_FRONTEND_BASE)

LABELS = {1: 'Tiger', 2: 'Elephant', 3: 'Jaguar', 4: 'Human'}
unknownCounter = 0

inMemoryPerClassDb = defaultdict(list)

def associateIdentities(detections, devName):
    if not isinstance(detections, Iterable):
        return []

    identities = []
    # For each identity, find closest match and return
    for det in detections:
        classid = int(det.classid)
        if classid not in LABELS:
            LABELS[classid] = 'Unknown{}'.format(unknownCounter)
            unknownCounter += 1

        detName = '{}_{}'.format(devName, LABELS[classid])
        if not isinstance(det, DetectionResult):
            print('Malformed entry {}, failing operation...'.format(det))
            return []

        currentDetections = inMemoryPerClassDb[detName]
        if len(currentDetections) == 0:
            inMemoryPerClassDb[detName].append(det)
            # First detection, assign label
            name = '{}_{}_{}'.format(devName, LABELS[classid], len(inMemoryPerClassDb[detName]))
        else:
            # More than one existing detection, find correct one
            # Find greatest IOU overlap. If < 0.25, new identity. If > 75%, same identity
            iou_overlaps = [iou(det.bounding_box, x.bounding_box) for x in currentDetections]
            if max(iou_overlaps) < 0.25: # New id
                inMemoryPerClassDb[detName].append(det)
                name = '{}_{}_{}'.format(devName, LABELS[classid], len(inMemoryPerClassDb[detName]))
            else:
                index = np.argmax(iou_overlaps)
                inMemoryPerClassDb[index] = det
                name = '{}_{}_{}'.format(devName, LABELS[classid], index + 1)
        identities.append(name)


    return identities
                

class DeviceMonitor:
    def __init__(self, name, centralMonitor, imagePublisher, timeout = 10):
        self.name = name
        self.active_tracks = dict() # Per class object tracker lists
        self.timeout = timeout # seconds
        self.centralMonitor = centralMonitor
        self.monitor = SubjectMonitor(timeout)
        self.imagePublisher = TimedImagePublisher(imagePublisher)

    def __repr__(self):
        return self.name

    def onEvent(self, client, userdata, msg):
        #print('Got message for device: {}: {}'.format(self.name, msg.payload.decode()))
        if msg.topic.decode().split('/')[-1] == 'image':
            self.imagePublisher.publishImage(msg.payload)
            return

        detections = eval(msg.payload.decode())

        if not isinstance(detections, Iterable):
            print('Invalid type provided in event, expected iterable, got: {}'.format(type(detections)))
            return

        identities = associateIdentities(detections, self.name)
        if not len(identities) == len(detections):
            print('Data malformed, id association failed')
            return
        
        for det,id in zip(detections, identities):
            self.addDetection(det, id, len(detections))

    def addDetection(self, detection, idf, total):
        self.monitor.addDetection(detection, idf)
        self.centralMonitor.addDetection(self.name, detection, idf, total)

    def getActiveTracks(self):
        return self.monitor.getActiveDetections()
 
class SubjectMonitor:
    def __init__(self, timeout=3000):
        self.detections = dict()
        self.timeout = timeout

    def addDetection(self, detection, idf):
        curtime = time.time()
        if idf is None:
            print('Received invalid idf')
            return

        if not idf in self.detections:
            print('Starting new track for id: {}'.format(idf))
            self.detections[idf] = TrackedSubject(Tracker(detection.bounding_box), curtime)
        else:
            print('Found existing id as: {}'.format(idf))
            tmp = self.detections[idf]
            tmp.tracker.addFrame(detection.bounding_box)
            self.detections[idf] = TrackedSubject(tmp.tracker, curtime)

        return idf

    def getLatestTracklet(self, idf):
        if idf in self.detections and len(self.detections[idf].tracker.tracklets) > 0:
            return self.detections[idf].tracker.tracklets[-1]
        return None

    def getActiveDetections(self):
        # Remove stale tracklets and return all active detections
        curtime = time.time()
        # Remove inactive tracks
        self.detections = {k:v for k,v in self.detections.items() if (curtime - v.timestamp) < self.timeout}

        return {k:(v.tracker.getTracklets(), v.timestamp) for k,v in self.detections.items()} 

class CrossCameraMonitor:
    def __init__(self, feClient, timeout=3000):
        self.registered = dict()
        self.feClient = feClient
        self.timeout = timeout
        self.inCounter = defaultdict(int)
        self.DETER_TIMEOUT = 6
        self.timerSet = defaultdict(bool)
        self.AREA_THRESH = 0.3 * (300 * 300) # Covers 50% of the image
        self.staticMap = defaultdict(bool)
        self.wasAlerted = defaultdict(bool)

    def sendFeDetectionMessage(self, device, numDetections, deter):
        channel = '{}/{}'.format(MQTT_FRONTEND_BASE, device)
        data = DetectionInfo(device, numDetections, deter)
        self.feClient.message(channel, json.dumps(data._asdict()))

    def addRegistration(self, name):
        if name in self.registered:
            print('{} already registered, ignoring...')
        else:
            self.registered[name] = SubjectMonitor(self.timeout)
            # Send message to frontend
            self.feClient.message(MQTT_FRONTEND_REG, name)

    def addDetection(self, name, detection, idf, numDetections):
        if name in self.registered:
            deterState = None 
            self.registered[name].addDetection(detection, idf)
            track = self.registered[name].getLatestTracklet(idf)
            if track is None:
                return

            if track.zdir == 'i':
                self.inCounter[idf] += 1
                if not self.timerSet[idf]:
                    self.timerSet[idf] = True
                # Start timer in case there were no detections
                    timer = threading.Timer(self.DETER_TIMEOUT, self.timerFunc, [name, idf])
                    timer.start()
                    self.staticMap[idf] = False
            elif track.zdir == 'o':
                self.inCounter[idf] = 0
                if self.wasAlerted[idf]:
                    deterState = 'success'
                self.staticMap[idf] = False
                self.wasAlerted[idf] = False
            else:
                self.staticMap[idf] = True

            if self.inCounter[idf] > 2 or Tracker([])._getBboxArea(detection.bounding_box) > self.AREA_THRESH:
                print('Glowing LEDS!')
                print('Glowing LEDS!')
                print('Glowing LEDS!')
                deterState = 'alert'
                self.wasAlerted[idf] = True
            self.sendFeDetectionMessage(name, numDetections, deterState)
        else:
            print('No registered device for {}'.format(name))

    def timerFunc(self, name, idf):
        if self.inCounter[idf] > 0 and not self.staticMap[idf]:
            print()
            print('!!!!!!!!!!!!!!!!')
            print('Deterring animal {} failed, sending SMS'.format(idf))
            print('!!!!!!!!!!!!!!!!')
            print()
            self.timerSet[idf] = False
            self.sendFeDetectionMessage(name, 1, 'failed')

        elif self.staticMap[idf]:
            print()
            print('Watching animal {} for some more time...'.format(idf))
            print()
            self.staticMap[idf] = False
            threading.Timer(self.DETER_TIMEOUT, self.timerFunc, [name, idf]).start()
            self.timerSet[idf] = True
            self.sendFeDetectionMessage(name, 1, 'alert')
        else:
            print()
            print('Animal {} successfully deterred...'.format(idf))
            print()
            self.timerSet[idf] = False
            self.sendFeDetectionMessage(name, 0, 'success')

    def getActiveDetections(self):
        # Combine tracklets from all registered device, sort them by time first
        results = defaultdict(list)
        for name,reg in self.registered.items():
            dets = reg.getActiveDetections()
            for k,v in dets.items():
                # Index the detections by classid, across all cameras
                results[k].append((name, v[0], v[1]))

        return results
   
class CentralServer:
    def __init__(self, server, port=1883, feport=9001):
        self.subscriber = MqttSubscriber(server, port)
        self.topic_funcs = {MQTT_REG_PATH : self.register, MQTT_REFRESH: self.refresh}
        self.registrations = []
        self.frontend = MqttWebsocketClient(server, feport);
        self.monitor = CrossCameraMonitor(self.frontend)
        self.detHistoryTimeout = 10 # 10 seconds
        threading.Timer(self.detHistoryTimeout, self.sendDetectionHistory, []).start()

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
                client.subscribe('{}/image'.format(parts[1]))
                reg = DeviceMonitor(parts[0], self.monitor)
                self.registrations.append(reg)
                self.topic_funcs[parts[1]] = reg.onEvent
                self.monitor.addRegistration(parts[0])

    def sendDetectionHistory(self):
        results = []
        for idf, tracks in self.monitor.getActiveDetections().items():
            if not len(tracks) > 0:
                continue
            # Pick latest device only
            det = tracks[-1]
            time = int(det[2] * 1000) # Conver to ms since epoch
            tr = ''
            if len(det[1]) > 0:
                tr = getDirections([det[1][-1]])[0]
            results.append(DetectionHistory(idf, det[0], tr, time)._asdict())

        self.frontend.message(MQTT_FRONTEND_DET_HISTORY, json.dumps(results))
        threading.Timer(self.detHistoryTimeout, self.sendDetectionHistory, []).start()


    def refresh(self, client, userdata, msg):
        print()
        print('-----------------')
        print('Current active devices: {}'.format(self.registrations))
        for reg in self.registrations:
            tracks = reg.getActiveTracks()
            print('Total {} active tracks found for device: {}'.format(len(tracks), reg))
            for t,v in tracks.items():
                # print('Found Total entities with class id {}: {}'.format(t, len(v)))
                print('Movements of {}: {}'.format(t, getDirections(v[0])))

        print()
        print('History of tracks:')
        for idf,tracks in self.monitor.getActiveDetections().items():
            print('Identity: {}, Movement History: {}'.format(idf, [(x[0], getDirections(x[1]), x[2]) for x in tracks]))

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
            print(traceback.print_exc())

    def onConnect(self, client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        client.subscribe(MQTT_REG_PATH)
        client.subscribe(MQTT_REFRESH)

    def run(self):
        self.subscriber.run(self.onConnect, self.on_message)        

if __name__ == '__main__':
    server = CentralServer(MQTT_SERVER)
    server.run()
