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

publish.single(channel, message, hostname=server, port=9001, transport='websockets')
