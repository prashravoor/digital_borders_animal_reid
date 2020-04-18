import paho.mqtt.client as mqtt

class MqttSubscriber:
    def __init__(self, server, port = 1833):
        self.client = mqtt.Client()
        self.server = server
        self.port = port

    def run(self, onConnect, onMessage):
        self.client.on_connect = onConnect
        self.client.on_message = onMessage
        self.client.connect(self.server, self.port, 60)
        self.client.loop_forever()

