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

### Running with MobileNetV2-SSD
Download the model file from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz) ([source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)) and save them under the ssd folder. You can then run the `animal_detection.py` and the `mobilenet_ssd.ipynb` files using the saved model.

## Feature Extraction and SVM Regression
In order to test the feature extraction and clustering using SVM, the following are required: <br>
* MongoDB - Install using `sudo apt install mongod`
* pymongo - Install using `python -m pip install pymongo`
* Model Weights (Pre-Trained over ImageNet)
* * AlexNet - Download from [here](dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel)
* * GoogLeNet - Download from [here](dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
* * ResNet50 - Download the .caffemodel for ResNet50 from [here](https://onedrive.live.com/?authkey=%21AAFW2%2DFVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
* Save all the models with the correct names under the `models` folder as follows: <br>
* * AlexNet - alexnet.caffemodel <br>
* * GoogLeNet - googlenet.caffemodel <br>
* * ResNet50 - resnet50.caffemodel <br>
<br>
* Ensure that MongoDB is up, and is listening on `localhost:27017`
<br>

### Setting up the data
Store all the images into an appropriate folder. The folder needs to have a file called `class_mapping.txt`, which contains the image file name and the class label as follows: <br>
`Image File Prefix  Individual Id` <br>
The file for `ELPephants` contains this information as requried, but the AMUR dataset does not. <br>
Generate the file for AMUR as needed, and put int under the data folder. <br>
A sample image dataset is provided under `amur_small` <br>

### Running the code
The Jupyter-Notebook file is added for quick tests. Otherwise, the python program can be run as: <br>
`python extract_features_and_store_to_mongo.py amur_small` <br>
Replace **amur_small** with the path to the correct image dataset as needed <br>
By default, the program will evaluate all the datasets provided, under all three models specified, and save the output of each layer of the network into the DB <br>


### Partitioning images for Train and Test during Re-Identification training
Since open-set identification is being tested, it is required to remove some of the known ids and train the SVM using the reamining ids. <br>
(For AMUR database only) First, using the `reid_list_train.csv` file, only those images for which the id is known is retained inside the `train` folder <br>
Of those images retained in the `train` folder, randomly select 10 ids, and move every image labelled for these 10 ids into a new folder called `test`. The list of the ids that have been moved are stored under `test/reid_list_test.csv` <br>
Since a subset of the database is again needed to test the effectiveness of the SVM classifier, for every id that has not been removed (i.e. ids that are in `reid_list_train` but not in `reid_list_test`), remove 25% of the images and move them under the `test` folder. <br>
<br>
The SVM can then be trained using the `train` subset, and their generalization accuracy is measured over the `test` data. <br>
<br>
<br>
For the ELP dataset, the train and validation splits are already provided under `train.txt` and `val.txt`. These are directly used to partition the dataset. <br>
