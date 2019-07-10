#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

for feature in rgb audio all ; do
    echo
    echo "FINE TUNING RESNET MODEL ON $feature FEATURES...."
    echo
    for i in 512 1024 2048 4096 ; do
        echo
        echo "TRAINING MODEL: UNITS ($i, $i, $i) FEATURE ($feature)....."
        echo
        python train_inference.py -m resnet -u "$i, $i, $i" -f ${feature}
    done
done
