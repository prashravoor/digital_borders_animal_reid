#!/bin/bash

# set base as necessary
BASE=`pwd`

export PYTHONPATH=$PYTHONPATH:$BASE:"$BASE/raspberry_pi/mqtt":"$BASE/raspberry_pi/object_tracking"
