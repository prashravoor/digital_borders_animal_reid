#!/bin/bash

# Vary samples per id, keeping number of ids constant
function varySamples() {
    modelPath=$1
    imgFolder=$2
    numIds=$3

    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=$numIds --samplesPerId=3
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=$numIds --samplesPerId=4
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=$numIds --samplesPerId=5
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=$numIds --samplesPerId=6
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=$numIds --samplesPerId=7
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=$numIds --samplesPerId=8

}


# Vary number of ids, keeping number of samples per id constant
function varyNumIds() {
    modelPath=$1
    imgFolder=$2
    numSamples=$3

    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=3 --samplesPerId=$numSamples
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=5 --samplesPerId=$numSamples
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=10 --samplesPerId=$numSamples
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=15 --samplesPerId=$numSamples
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=20 --samplesPerId=$numSamples
    python3 test_svm_model.py --modelPath=$modelPath --imgFolder=$imgFolder --numIds=25 --samplesPerId=$numSamples
}

# Jaguars
varySamples svm_models_trained/jaguar-alexnet-relu6-linear-pca.model jaguars/reid 10
varyNumIds svm_models_trained/jaguar-alexnet-relu6-linear-pca.model jaguars/reid 5
