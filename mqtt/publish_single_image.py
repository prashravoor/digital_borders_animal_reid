import paho.mqtt.publish as publish
import sys
import base64

args = sys.argv
if not len(args) == 4:
    print('Usage: cmd <server ip> <channel> <message>')
    exit()

server = args[1]
channel = args[2]
message = args[3]

filename = '/tmp/amur.jpg'

#publish.single(channel, message, hostname=server, port=9001, transport='websockets')
with open(filename, 'rb') as f:
    data = base64.b64encode(f.read())

publish.single(channel, data, hostname=server, port=9001, transport='websockets')
