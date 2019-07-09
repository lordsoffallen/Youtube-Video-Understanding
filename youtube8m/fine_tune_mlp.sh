#!/usr/bin/env bash


for feature in rgb audio all ; do
    echo
    echo "FINE TUNING MLP ON $feature FEATURES...."
    echo
    for i in 512 1024 2048 4096 ; do
        echo "Training model with $i unit parameters....."
        python train_inference.py -m mlp -u "$i" -f ${feature} -l binary

        echo "Training model with $i, $i unit parameters....."
        python train_inference.py -m mlp -u "$i, $i" -f ${feature} -l binary

        echo "Training model with $i, $i, $i unit parameters....."
        python train_inference.py -m mlp -u "$i, $i, $i" -f ${feature} -l binary
    done
done

