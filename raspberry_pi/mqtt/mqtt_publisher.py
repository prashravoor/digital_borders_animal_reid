import paho.mqtt.publish as publish

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
