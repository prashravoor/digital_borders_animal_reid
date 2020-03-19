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

# Raspberry Pi Code
## Setup
The official TFLite provided by Google is rather slow. Install the alternative compiled version provided by PINTO as mentioned in the corresponding readme
Install MQTT using `python3 -m pip install mqtt` <br>
On a laptop / desktop, setup the MQTT broker service using `sudo apt install mosquitto` <br>
Provide values for the MQTT server in the Raspberry Pi code files, and run the program <br>
Before running any of the MQTT programs, setup the PYTHONPATH correctly using `export PYTHONPATH=$PYTHONPATH:<Repo base dir>` <br>

## Tracking
Run the subscriber program on the laptop using `python3 raspberry_pi/mqtt/subscriber.py` to start the listening service for the tracking

# Identification Pipeline with Triplet loss
## Setup
Install the following required packages for using the Identification pipeline by running `python3 -m pip install -r triplet_loss_req.txt` <br>

## Testing model accuracy
Once the model has been trained (using the open-reid strong baseline repo), copy the model and convert it to an ONNX model using <br>
`bash convert_torch_to_onnx.sh <input model path> <output model path>` <br>
Conversion to ONNX is required since PyTorch on CPU is extremely slow. <br>
Run the code to test feature extraction accuracy and show the TSNE graphs using `python3 test_triplet_loss.py <image folder> <ONNX model path>`

## Training Using Re-id Strong Baseline
Download the Open-Reid code base from [here] (https://github.com/Cysu/open-reid.git) <br>
Apply the changes as required using the patch files stored in `patches/open_reid` folder <br>
Download the Re-id strong baseline code from [here](https://github.com/michuanhaohao/reid-strong-baseline.git) <br>
Apply the changes as required using the patch files stored in `patches/reid_strong_baseline` folder <br>

### Prepare the dataset
Multiple scripts are needed to be run before the dataset can be used for reid-strong baseline. All required scripts are listed below. Run them in the same order. The sample assumes images are stored in a folder named `ELPephants\reid_faces`. The metdatafile `class_mapping.txt` which contains each file to class id mapping is required to be present in the same folder<br>
```bash
python3 create_open_reid_splits.py ELPephants\reid_faces ELPephants\faces_open_reid # Creates two folders, train and test
python3 rename_images_to_int_names.py ELPephants\faces_open_reid\train # Renames all files to integer names as needed by Open Re-id
python3 remap_labels_contiguous.py ELPephants\faces_open_reid\train # If there are any missing identities, replace them with continuous ids
# Move to Open Reid code base, and start the run and stop it once the datasets are created. It creates the images, splits.json and meta.json files
# ...
python3 partition_ds_for_open_reid.py ..\open_reid\amur_data\elp ..\reid-strong-baseline\data\elp # Optional split number between [0,10] can also be specified
```

### Training
From the reid-strong-baseline folder, run the appropriate training file by specifying the requried config file <br>
For example: `train.bat softmax_triplet_with_center_elp.yml`

### Testing
Once the training is completed, test the performance over the test data <br>
`python3 test_model_reid.py ..\reid-strong-baseline\elp_test\resnet50_model_100.pth ELPephants\faces_open_reid\test ..\reid-strong-baseline\configs\elp_softmax_triplet_with_center.yml
