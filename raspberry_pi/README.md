# Object Detection and Animal Re-Identification on Raspberry PI

## Installation
Install the Tensorflow lite build from PITO using this [wheel](https://github.com/PINTO0309/TensorflowLite-bin/raw/master/1.15.0/tflite_runtime-1.15.0-cp35-cp35m-linux_armv7l.whl). It's a little faster than the original TFLite and also exposes a few more native APIs (like `set_num_threads`. Switch to the official version once they support them.<br>
Full install instructions for TFLite can be found [here](https://github.com/PINTO0309/TensorflowLite-bin#tensorflowlite-bin) <br>
Any additional dependencies, such as Numpy and PiCamera need to be installed using Pip. Note: Numpy Version `1.12` works, `1.18` failed, most likely due to OS incompatibility. <br>

## Running the Detector
Once all dependencies are installed, setup the PiCamera module appropriately. Download the quantized model for TFLite. This repo uses a specially trained quantized model, but a TFLite model pretrained on COCO dataset can be found [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) <br>

Run using `python3 object_tracking/run_object_detection.py <path to tflite model> <MQTT broker IP> [display camera images>` <br>
Each frame processing takes around 350-400ms on average. <br>
In order to effectively track, ensure that the `subscriber` program is running on the same server as the MQTT broker <br>

## Setting up the MQTT server
Angular and other JS frameworks work with MQTT only with web-sockets, and not using regular TCP connection. To enable communication between the local subscriber and the frontend application, add the following to a file (create if needed) `/etc/mosquitto/conf.d/default.conf`: <br>
```bash
listener 1883
protocol mqtt

listener 9001
protocol websockets
```

Restart the mosquitto service through `sudo service mosquitto restart`. The python client which listens over MQTT will need to relay the required messages over websockets to the Frontend <br>
If needed, the front-end can be changed to a standalone application, or some other mechanism can be used instead of MQTT b/w subscriber and frontend <br>
