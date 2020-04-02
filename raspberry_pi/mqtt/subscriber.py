import paho.mqtt.client as mqtt
import time
from object_tracker import Tracker, Tracklet, mergeTracklets, getDirections
from im_utils import drawBoundingBoxLabel
from collections import namedtuple, Iterable, defaultdict
from utils import DetectionResult, BoundingBox
from mqtt_subscriber import MqttSubscriber
from mqtt_publisher import MqttWebsocketClient,ImageWsPublisher,TimedImagePublisher
import threading
import json
import numpy as np
import traceback
from id_association import IdAssociator, FeatureVectorDatabase
from feature_extractor import FeatureExtractor
import sys
import cv2
import pickle

TrackedSubject = namedtuple('TrackedSubject', 'tracker timestamp')
DetectionInfo = namedtuple('DetectionInfo', 'name activeDetections deterState')
DetectionHistory = namedtuple('DetectionHistory', 'name lastSeen travelDirection lastSeenTime')

MQTT_SERVER = "localhost"
MQTT_REG_PATH = "registration"
MQTT_REFRESH = 'refresh'
MQTT_CLEAR = 'clear'
MQTT_FRONTEND_BASE = 'frontend'
MQTT_FRONTEND_REG = '{}/registration'.format(MQTT_FRONTEND_BASE)
MQTT_FRONTEND_DET_HISTORY = '{}/detectionHistory'.format(MQTT_FRONTEND_BASE)

LABELS = {0: 'Human', 1: 'Tiger', 2: 'Elephant', 3: 'Jaguar', 4: 'Human'}

class SubjectMonitor:
    def __init__(self, idAssociator, timeout=3000):
        self.detections = dict()
        self.timeout = timeout
        self.idAssociator = idAssociator

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

    def clearDetections(self):
        self.detections = dict()

class CrossCameraMonitor:
    def __init__(self, feClient, idAssociator, timeout=3000):
        self.registered = dict()
        self.DETER_STATES = {0: 'None', 1: 'alert', 2: 'success', 3: 'failed', 4: 'same'}
        self.feClient = feClient
        self.timeout = timeout
        self.inCounter = defaultdict(int)
        self.outCounter = defaultdict(int)
        self.DETER_TIMEOUT = 5
        self.AREA_THRESH = 0.3 * (300 * 300) # Covers 50% of the image
        self.COUNT_THRESH = 3 # 2 continous frames to confirm movement direction
        self.staticMap = defaultdict(bool)
        self.wasAlerted = defaultdict(bool)
        self.idAssociator = idAssociator
        self.detHistoryTimeout = 1 # 10 seconds
        self.lock = threading.Lock()
        self.imagePublishers = dict()
        self.threadMap = dict()
        self.activeDetectionMap = dict()
        self.activeDetectionLock = threading.Lock() 
        self.threadid = 0
        self.threadMapLock = threading.Lock()
        self.ACTIVE_DET_TIMEOUT = 3
        self.prev_deterState = 0


        th = threading.Timer(self.detHistoryTimeout, self.sendDetectionHistory, [])
        th.setDaemon(True)
        th.start()

    def sendFeDetectionMessage(self, device, numDetections, deter):
        channel = '{}/{}'.format(MQTT_FRONTEND_BASE, device)
        if deter == self.DETER_STATES[0]:
            deter = None
        data = DetectionInfo(device, numDetections, deter)
        self.feClient.message(channel, json.dumps(data._asdict()))

    def addRegistration(self, name):
        if name in self.registered:
            print('{} already registered, ignoring...')
        else:
            self.registered[name] = SubjectMonitor(self.idAssociator, self.timeout)
            self.imagePublishers[name] = TimedImagePublisher(
                    ImageWsPublisher('{}/{}'.format(MQTT_FRONTEND_BASE, name),
                                      self.feClient.server, self.feClient.port), 0.001)
            # Send message to frontend
            self.feClient.message(MQTT_FRONTEND_REG, name)

    def detectionAdded(self, name, idf, detection):
        deterState = 0
        if name in self.registered:
            track = self.registered[name].getLatestTracklet(idf)
            if track is None:
                return deterState

            if track.zdir == 'i':
                self.inCounter[idf] += 1
                self.outCounter[idf] = 0

                self.threadMapLock.acquire()
                if idf in self.threadMap:
                    # Start timer in case there were no detections
                    # Stop any active threads
                    print('Existing thread found, deactivating...')
                    self.threadMap[idf][1].cancel()
                    del self.threadMap[idf]

                timer = threading.Timer(self.DETER_TIMEOUT, self.timerFunc, [name, idf, self.threadid])
                self.threadMap[idf] = (self.threadid, timer)  # Overwrite event
                self.threadid += 1
                self.threadMapLock.release()

                timer.setDaemon(True)
                timer.start()
                self.staticMap[idf] = False

            elif track.zdir == 'o':
                self.outCounter[idf] += 1
                self.inCounter[idf] = 0 
                if self.wasAlerted[idf] and self.outCounter[idf] > self.COUNT_THRESH:
                    deterState = 2
                self.staticMap[idf] = False
                self.wasAlerted[idf] = False
            else:
                self.staticMap[idf] = True

            area = Tracker([])._getBboxArea(detection.bounding_box)
            if self.inCounter[idf] > self.COUNT_THRESH or self.wasAlerted[idf]:
            #if area > self.AREA_THRESH:
                print('Glowing LEDS!')
                print('Glowing LEDS!')
                print('Glowing LEDS!')
                deterState = 1
                self.wasAlerted[idf] = True
        else:
            print('No registered device for {}'.format(name))

        return deterState

    def timerFunc(self, name, idf, thid):
        self.threadMapLock.acquire()
        if idf in self.threadMap:
            if not self.threadMap[idf][0] == thid:
                self.threadMapLock.release()
                print('Been replaced by another thread...')
                return # Replaced by another thread
            del self.threadMap[idf]
        else:
            print('Device not in map, exiting...')
            self.threadMapLock.release()
            return
        self.threadMapLock.release()

        if self.inCounter[idf] > 1 and not self.staticMap[idf]:
            print()
            print('!!!!!!!!!!!!!!!!')
            print('Deterring animal {} failed, sending SMS'.format(idf))
            print('!!!!!!!!!!!!!!!!')
            print()
            self.sendFeDetectionMessage(name, 1, 'failed')
            self.inCounter[idf] = 0
        elif self.staticMap[idf]:
            print()
            print('Watching animal {} for some more time...'.format(idf))
            print()
            self.staticMap[idf] = False
            th = threading.Timer(self.DETER_TIMEOUT, self.timerFunc, [name, idf, thid])
            self.threadMapLock.acquire()
            if idf not in self.threadMap: # If there is an entry, another thread has replaced this one, so remove it
                self.threadMap[idf] = (thid, th)
                th.setDaemon(True)
                th.start()
            self.threadMapLock.release()
            self.sendFeDetectionMessage(name, 1, 'alert')
        elif self.outCounter[idf] > 1: 
            print()
            print('Animal {} successfully deterred...'.format(idf))
            print()
            self.sendFeDetectionMessage(name, 1, 'success')
            self.outCounter[idf] = 0
        else:
            print()
            print('Resetting to normal...')
            print()
            #self.sendFeDetectionMessage(name, 0, 'None')

    def sendNoActiveDetectionsMessage(self, name):
        self.activeDetectionLock.acquire()
        if name in self.activeDetectionMap:
            del self.activeDetectionMap[name]
            self.sendFeDetectionMessage(name, 0, 'None')
        self.activeDetectionLock.release()

    def getActiveDetections(self):
        # Combine tracklets from all registered device, sort them by time first
        results = defaultdict(list)
        for name,reg in self.registered.items():
            dets = reg.getActiveDetections()
            for k,v in dets.items():
                # Index the detections by classid, across all cameras
                results[k].append((name, v[0], v[1]))

        return results

    def getRegisteredDevices(self):
        return list(self.registered.keys())

    def sendDetectionHistory(self, periodic=True):
        results = []
        for idf, tracks in self.getActiveDetections().items():
            if not len(tracks) > 0:
                continue
            # Pick latest device only
            det = tracks[-1]
            time = int(det[2] * 1000) # Conver to ms since epoch
            tr = ''
            if len(det[1]) > 0:
                tr = getDirections([det[1][-1]])[0]
            results.append(DetectionHistory(idf, det[0], tr, time)._asdict())

        self.feClient.message(MQTT_FRONTEND_DET_HISTORY, json.dumps(results))
        if periodic:
            th = threading.Timer(self.detHistoryTimeout, self.sendDetectionHistory, [])
            th.setDaemon(True)
            th.start()

    def sendImage(self, name, image, dets, ids):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(len(dets)):
            image = drawBoundingBoxLabel(image, ids[i], dets[i])

        self.imagePublishers[name].publishImage(cv2.imencode('.jpg', image)[1])

    def receivedDetection(self, device, detectionMap):
        # Map has detections, image keys
        try:
            self.lock.acquire()
            detections = eval(detectionMap['detections'])
            # image = np.array(eval(detectionMap['image'])).astype(np.float32)
            image = np.frombuffer(detectionMap['image'], dtype='uint8')
            image = cv2.imdecode(image, flags=1)
            identities = self.idAssociator.update(device, image, detections)
            identities = ['{}_{}'.format(LABELS[int(detections[i].classid)], identities[i]) for i in range(len(identities))]
            deterState = 0
            monitor = self.registered[device]
            for i in range(len(identities)):
                monitor.addDetection(detections[i], identities[i])
                tmp = self.detectionAdded(device, identities[i], detections[i])
                deterState = max(deterState, tmp)

            self.sendFeDetectionMessage(device, len(detections), self.DETER_STATES[deterState])
            self.sendImage(device, image, detections, identities)

            timer = threading.Timer(self.ACTIVE_DET_TIMEOUT, self.sendNoActiveDetectionsMessage, [device])
            timer.setDaemon(True)

            self.activeDetectionLock.acquire()
            if device in self.activeDetectionMap:
                self.activeDetectionMap[device].cancel() # Cancel any outstanding thread
                del self.activeDetectionMap[device]
            self.activeDetectionMap[device] = timer
            self.activeDetectionLock.release()

            timer.start()

        finally:
            self.lock.release()

    def refresh(self, client, userdata, msg):
        print()
        print('-----------------')
        print('Current active devices: {}'.format(self.registered))
        for reg in self.registered:
            tracks = self.registered[reg].getActiveDetections()
            print('Total {} active tracks found for device: {}'.format(len(tracks), reg))
            for t,v in tracks.items():
                # print('Found Total entities with class id {}: {}'.format(t, len(v)))
                print('Movements of {}: {}'.format(t, getDirections(v[0])))

        print()
        print('History of tracks:')
        for idf,tracks in self.getActiveDetections().items():
            print('Identity: {}, Movement History: {}'.format(idf, [(x[0], getDirections(x[1]), x[2]) for x in tracks]))

        print('-----------------')
        print()

        # Also write data to frontend
        for k in self.getRegisteredDevices():
            self.feClient.message(MQTT_FRONTEND_REG, k)

        self.sendDetectionHistory(periodic=False)

    def clearDetections(self, client, userdata, msg):
        for _,reg in self.registered.items():
            reg.clearDetections()
 
class CentralServer:
    def __init__(self, server, idAssociator, port=1883, feport=9001):
        self.subscriber = MqttSubscriber(server, port)
        self.frontend = MqttWebsocketClient(server, feport)
        self.monitor = CrossCameraMonitor(self.frontend, idAssociator)
        self.server = server
        self.fePort = feport
        self.topic_funcs = {MQTT_REG_PATH : self.register, MQTT_REFRESH: self.monitor.refresh, MQTT_CLEAR: self.monitor.clearDetections}

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
                self.monitor.addRegistration(parts[0])
                self.topic_funcs[parts[1]] = self.receivedDetection

    def receivedDetection(self, client, userdata, msg):
        detection = pickle.loads(msg.payload)
        self.monitor.receivedDetection(msg.topic, detection)

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
        client.subscribe(MQTT_CLEAR)

    def run(self):
        self.subscriber.run(self.onConnect, self.on_message)        

if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 2:
        print('Usage: cmd modelpath')
        exit()

    modelpath = args[1]

    print('Loading feature extractor...')
    start = time.time()
    fe = FeatureExtractor(modelpath)
    fe.loadModel()
    print('Model Load time: {:.4f}s'.format(time.time() - start))
# Change to use appropriate models
    feMap = dict()
    for k,_ in LABELS.items():
        feMap[k] = fe

    print('Warming up model...')
    fe.extract(np.random.randint(0, 255, (fe.INPUT_HEIGHT, fe.INPUT_HEIGHT, 3)).astype(np.uint8))

    fedb = FeatureVectorDatabase()
    idAssociator = IdAssociator(feMap, fedb)

    server = CentralServer(MQTT_SERVER, idAssociator)
    server.run()
