#!/usr/bin/env bash

# FINE TUNE RESNET MODEL
################################################################

echo $'FINE TUNING RESNET MODEL ON RGB FEATURES....\n'

for i in 512 1024 2048 4096 ; do
    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m resnet -u '$i, $i, $i' -f rgb
done

echo $'FINE TUNING RESNET MODEL ON AUDIO FEATURES....\n'

for i in 512 1024 2048 4096 ; do
    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m resnet -u '$i, $i, $i' -f audio
done

echo $'FINE TUNING RESNET MODEL ON RGB w/AUDIO FEATURES....\n'

for i in 512 1024 2048 4096 ; do
    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m resnet -u '$i, $i, $i' -f all
done

################################################################

# FINE TUNE MLP MODEL
################################################################

echo $'FINE TUNING MLP MODEL ON RGB FEATURES....\n'

for i in 512 1024 2048 ; do
    echo $'Training model with $i unit parameters.....'
    python train_inference.py -m mlp -u '$i' -f rgb -l binary

    echo $'Training model with $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i' -f rgb -l binary

    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i, $i' -f rgb -l binary
done

echo $'FINE TUNING MLP MODEL ON AUDIO FEATURES....\n'

for i in 512 1024 2048 ; do
    echo $'Training model with $i unit parameters.....'
    python train_inference.py -m mlp -u '$i' -f audio -l binary

    echo $'Training model with $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i' -f audio -l binary

    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i, $i' -f audio -l binary
done

echo $'FINE TUNING MLP MODEL ON RGB w/AUDIO FEATURES....\n'

for i in 512 1024 2048 ; do
    echo $'Training model with $i unit parameters.....'
    python train_inference.py -m mlp -u '$i' -f all -l binary

    echo $'Training model with $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i' -f all -l binary

    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i, $i' -f all -l binary
done

################################################################


# FINE TUNE MoE MODEL
################################################################

echo $'FINE TUNING MIXTURE OF EXPERTS MODEL ON RGB FEATURES....\n'

for i in 512 1024 2048 ; do
    # TODO See if adding a unit helps. If not iterate over experts
    echo $'Training model without unit parameters.....'
    python train_inference.py -m moe -f rgb -l binary --num_experts 2

    echo $'Training model with $i unit parameters.....'
    python train_inference.py -m moe -u '$i' -f rgb -l binary --num_experts 2
done

echo $'FINE TUNING MIXTURE OF EXPERTS MODEL ON AUDIO FEATURES....\n'

for i in 512 1024 2048 ; do
    echo $'Training model with $i unit parameters.....'
    python train_inference.py -m moe -u '$i' -f audio -l binary

    echo $'Training model with $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i' -f audio -l binary

    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i, $i' -f audio -l binary
done

echo $'FINE TUNING MIXTURE OF EXPERTS MODEL ON RGB w/AUDIO FEATURES....\n'

for i in 512 1024 2048 ; do
    echo $'Training model with $i unit parameters.....'
    python train_inference.py -m mlp -u '$i' -f all -l binary

    echo $'Training model with $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i' -f all -l binary

    echo $'Training model with $i, $i, $i unit parameters.....'
    python train_inference.py -m mlp -u '$i, $i, $i' -f all -l binary
done

################################################################