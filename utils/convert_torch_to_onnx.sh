#!/bin/bash

IN=$1
OUT=$2

if [ -z $1 ]; then
    echo Missing input model path
    exit
fi

if [ -z $2 ]; then
    echo Missing output model path
    exit
fi

python3 convert_torch_to_onnx.py $IN $OUT"_tmp"
python3 -m onnxsim $OUT"_tmp" $OUT
rm $OUT"_tmp"
