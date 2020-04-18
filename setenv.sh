#!/bin/bash

# set base as necessary
BASE=`pwd`

export PYTHONPATH=$PYTHONPATH:$BASE:"$BASE/utils":"$BASE/mqtt":"$BASE/raspberry_pi":"$BASE/train/truncated_dcnns":"$BASE/train/reid-strong-baseline":"$BASE/central_server/object_tracking"
