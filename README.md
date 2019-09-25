# Digital Borders - Animal Re-Identification

## Setup
**Required python-version:** 3.6+
Install all needed requirements for installation:
`sudo apt install python-opencv`

### Setting up YOLO
Follow all instructions to install YOLO from [here](https://pjreddie.com/darknet/yolo/). <br>
Compiling the source code is optional. However, the network configuration and weights are needed. <br>

## Running the code
In the `extract_bb.py`, edit the default arguments as necessary for the **network configuration file**, **weights file** and **classes file**. <br>
Run the code on an image through: `python extract_bb.py -i <Path to Image File>` <br>
