# Object Detection and Animal Re-Identification on Raspberry PI

## Installation
Install the Tensorflow lite build from PITO using this [wheel](https://github.com/PINTO0309/TensorflowLite-bin/raw/master/1.15.0/tflite_runtime-1.15.0-cp35-cp35m-linux_armv7l.whl). It's a little faster than the original TFLite and also exposes a few more native APIs (like `set_num_threads`. Switch to the official version once they support them.<br>
Full install instructions for TFLite can be found [here](https://github.com/PINTO0309/TensorflowLite-bin#tensorflowlite-bin) <br>
Any additional dependencies, such as Numpy and PiCamera need to be installed using Pip. Note: Numpy Version `1.12` works, `1.18` failed, most likely due to OS incompatibility. <br>

## Running the Detector
Once all dependencies are installed, setup the PiCamera module appropriately. Download the quantized model for TFLite. This repo uses a specially trained quantized model, but a TFLite model pretrained on COCO dataset can be found [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) <br>

Run using `python3 object_tracking/run_object_detection.py <path to tflite model>` <br>
Each frame processing takes around 350-400ms on average. <br>
