#!/usr/bin/env bash

for feature in rgb audio all ; do
    echo
    echo "FINE TUNING MIXTURE OF EXPERTS ON $feature FEATURES...."
    echo

    for experts in 2 4 6 8 10 12 14 16 ; do
        for loss in binary huber ; do
            echo "Training model with $loss loss....."
            if [[ "$loss" == binary ]]; then
                for i in 512 1024 2048 ; do
                    echo "Training model with $i unit parameters....."
                    python train_inference.py -m moe -u ${i} -f ${feature} -l ${loss} --num_experts ${experts}
                done
            else
                python train_inference.py -m moe -f ${feature} -l ${loss} --num_experts ${experts}
            fi
        done
    done
done
