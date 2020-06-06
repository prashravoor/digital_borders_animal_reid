# Digital Borders - Animal Re-Identification

## Setup
**Required python-version:** 3.6+
Install all needed requirements for installation:
`sudo apt install python-opencv`

### Setting up YOLO
Follow all instructions to install YOLO from [here](https://pjreddie.com/darknet/yolo/). <br>
Compiling the source code is optional. However, the network configuration and weights are needed. <br>

### Running with MobileNetV2-SSD
Download the model file from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz) ([source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)) and save them under the ssd folder. You can then run the `demo/object_detection.py` and the `notebooks/ObjectDetection.ipynb` files using the saved model.

## Training Object Detection Model
See details for Object Detection and Re-Id training for `reid-strong-baseline` under the datasets folder <br>

## Training -- Feature Extractors using Truncated DCNNs
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
Details on how to acquire various datasets are provided under the `datasets` folder. <br>
Store all the images into an appropriate folder. The folder needs to have a file called `class_mapping.txt`, which contains the image file name and the class label as follows: <br>
`Image File Prefix  Individual Id` <br>
The file for `ELPephants` contains this information as requried, but the AMUR dataset does not. <br>
Generate the file for AMUR as needed, and put int under the data folder. <br>
A sample image dataset is provided under `sample_images/amur_small` <br>

### Running the code
The Jupyter-Notebook file is added for quick tests. Otherwise, the python program can be run as: <br>
`python train/truncated_dcnns/find_best_svm_model.py models_bin/ssd/saved_model/ sample_images/amur_small/` <br>
Replace **amur_small** with the path to the correct image dataset(s) as needed <br> **This could take several hours to complete.** <br>
By default, the program will evaluate all the datasets provided, under all three models specified, and save the output of each layer of the network into the DB <br>
The best models are saved to the `svm_models_trained` folder. <br>
The demo for one of the completed models can be performed using the `svm_identifier.py` script. For example: <br>
`python demo/svm_identifier.py svm_models_trained/<model> models_bin/ssd/saved_model <image folder>` <br>

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
To perform this split, run the `utils/create_open_reid_splits.py` script on the desired folder.<br>
More details on how to format the data suitable for training with `reid-strong-baseline` can be found under the `datasets` folder. <br>

# Identification Pipeline with Triplet loss
Only a very minimal (and slightly out-of-date) information of training and evaluation is provided here. <br>
For a more thorough description of the datasets, setting up and training procedure, see [here](https://github.com/prashravoor/reid-strong-baseline.git).<br>

## Training Using Re-id Strong Baseline
Checkout the training code from [here](https://github.com/prashravoor/reid-strong-baseline.git) and follow the instructions to complete training.

### Prepare the dataset
Multiple scripts are needed to be run before the dataset can be used for reid-strong baseline. All required scripts are listed below. Run them in the same order. The sample assumes images are stored in a folder named `ELPephants\reid_faces`. The metdatafile `class_mapping.txt` which contains each file to class id mapping is required to be present in the same folder<br>
```bash
python3 create_open_reid_splits.py ELPephants/reid_faces ELPephants/faces_open_reid # Creates two folders, train and test inside `faces_open_reid`
python3 rename_images_to_int_names.py ELPephants/faces_open_reid/train # Renames all files to integer names as needed by Open Re-id
python3 remap_labels_contiguous.py ELPephants/faces_open_reid/train # If there are any missing identities, replace them with continuous ids
# Move to Open Reid code base, and start the run and stop it once the datasets are created. It creates the images, splits.json and meta.json files
# ...
python3 partition_ds_for_open_reid.py ../open_reid/amur_data/elp ../reid-strong-baseline/data/elp # Optional split number between [0,10] can also be specified
```

### Training
From the reid-strong-baseline folder, run the appropriate training file by specifying the requried config file <br>
For example: `train.bat softmax_triplet_with_center_elp.yml` <br>
Configurations are included in the `datasets/reid/configs` folder. <br>


### Testing
Once the training is completed, test the performance over the test data <br>
Set the PYTHONPATH to the `reid-strong-baseline` folder through: `export PYTHONPATH=$PYTHONPATH:<path to reid strong baseline folder>` <br>
`python3 test/test_triplet_loss.py ../reid-strong-baseline/elp_test/resnet50_model_100.pth ELPephants/faces_open_reid/test` <br>
Additionally, Closed and Open set accuracy can be tested using `test/calc_closed_set_acc.py` and `test/calc_open_set_acc.py` scripts <br>

### Object Tracking Demo
Once the model has been trained using `reid-strong-baseline`, it first needs to be converted to Keras. This can be done using `python utils/convert_torch_to_keras.py <torch model> <output keras model name>`. The converted model can then be provided to the Object Tracker for re-identification <br>

A demo can be run over a video too, by running `python demo/video_demo.py <Object Detection Model> <Re-ID model (keras)> <video path>`. <br>

# Live Demo
On both the laptop and the raspberry pi, run `source setenv.sh` from the base folder of the repo to setup required PYTHONPATHs <br>

## Tracking
Run the subscriber program on the laptop using `python3 central_server/start_server.py <Re-Id model>` to start the listening service for the tracking <br>

## Frontend server
Once the MQTT broker service is up and running, as is the central server code, the frontend can be started using Angular CLI. For installation instructions, refer to the README file under `frontend/monitoring_server`. <br>
Once Node JS and Angular (version 9) are installed, set the IP address for the MQTT broker in the `app.module.ts` file. Start the frontend through: <br>
```bash
cd frontend/monitoring_server
ng serve --host 0.0.0.0
```

## Raspberry Pi 
The official TFLite provided by Google is rather slow. Install the alternative compiled version provided by PINTO as mentioned in the corresponding readme under the `raspberry_pi` folder. <br>
Install MQTT using `python3 -m pip install mqtt` <br>
On a laptop / desktop, setup the MQTT broker service using `sudo apt install mosquitto` <br>
Provide values for the MQTT server in the Raspberry Pi code files, and run the program <br>
`python raspberry_pi/run_object_detector.py <Object detection model (tflite)> <MQTT broker IP> [display - optional]` <br>

