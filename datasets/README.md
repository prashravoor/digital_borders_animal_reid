# Datasets 

## Description
There are four different datasets used as part of this project: <br>
 * [Amur Tigers](https://cvwc2019.github.io/challenge.html)
 * [Elephants](https://www.inf-cv.uni-jena.de/Research/Datasets/ELPephants.html)
 * [FaceScrub](https://github.com/faceteam/facescrub)
 * Jaguars (Provided on request by [Prof. Marcella Kelly](http://www.mjkelly.info/)

## Contents
The datasets are not included here, but the metadata for each of them required for training are included. <br>
The folder is organized into `reid` and `detection` and includes information for each dataset for re-identification and object detection training respectively. <br>
<br>
The contents of each dataset include: <br>
 * `trainval.txt` - Files used for the training-validation (and corresponding class labels for re-id)
 * `configs` - Configurations necessary for training

### Object Detection
Object Detection uses fine-tuning to train the network to detect three species - tigers, jaguars and elephants. Two models are trained in this project - [YOLO](https://pjreddie.com/darknet/yolo/) and MobileNetv2-SSD <br>
The pre-trained models used for both can be downloaded here - [YOLO-tiny weights](https://pjreddie.com/media/files/yolov3-tiny.weights) and [SSD](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz). <br>
<br>
YOLO uses darknet. Procedure to train the network can be found [here](https://towardsdatascience.com/implementing-yolo-on-a-custom-dataset-20101473ce53) <br>
A fair tutorial for TensorFlow object detection training is found [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html). TensorFlow 1.14 was used to train the object detector. <br>

### Re-Identification
Re-Identification training uses PyTorch, and is performed with the help of two open-source code bases - [OpenReId](https://github.com/Cysu/open-reid) and [Reid-Strong-Baseline](https://github.com/michuanhaohao/reid-strong-baseline). Setup both code bases as described in their respective documentation.<br> <br>
Changes needed to be applied to both datasets are included as a path file as part of this repo under the `patches` folder. They can be applied using `git apply <patch>` from the respective repos. <br> <br>
Each dataset is first partitioned using the a train and test split, by using 20% of the identities and all their images into a separate set. This can be done quickly using the `create_open_reid_splits.py` script. It requires a metadata file `normalized_class_mapping.txt`, which contains lines in the format `<filename>.jpg\t<class id int>` for each file in the dataset. The file needs to be located inside the dataset folder itself. <br> <br>
`reid-strong-baseline` requires the image data to be in a particular format. The conversion from the existing image name conventions to the required format is done using the `open-reid` code. Simply start training, allow time for the code to create the data set, and stop the training. The generated dataset can be converted to `train`, `gallery` and `query` sets using the `partition_ds_for_open_reid.py` script included in this repo. The input is the folder containing the dataset as created by Open-ReId<br> <br>
The partitions created in the dataset are included inside the respective dataset folders under the `reid` folder. The same parition can be used or a different identity combination can be used for the training. <br>
Training configurations for each dataset are located in the `configs` directory. The paths need to be appropriately changed. <br>
