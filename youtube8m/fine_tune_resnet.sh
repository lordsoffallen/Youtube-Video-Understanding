#!/usr/bin/env bash

for feature in rgb audio all ; do
    echo
    echo "FINE TUNING RESNET MODEL ON $feature FEATURES...."
    echo
    for i in 512 1024 2048 4096 ; do
        echo "Training model with $i, $i, $i unit parameters....."
        python train_inference.py -m resnet -u "$i, $i, $i" -f ${feature}
    done
done
