#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

for feature in rgb audio all ; do
    echo
    echo "FINE TUNING MIXTURE OF EXPERTS ON $feature FEATURES...."
    echo

    for experts in 2 4 6 8 10 12 14 16 ; do
        for loss in binary huber ; do
            if [[ "$loss" == binary ]]; then
                for i in 512 1024 ; do
                    echo
                    echo "TRAINING MODEL: UNITS ($i) FEATURE ($feature) EXPERTS ($experts) LOSS ($loss)....."
                    echo
                    python train_inference.py -m moe -u ${i} -f ${feature} -l ${loss} --num_experts ${experts}
                done
            else
                echo
                echo "TRAINING MODEL: UNITS (None) FEATURE ($feature) EXPERTS ($experts) LOSS ($loss)....."
                echo
                python train_inference.py -m moe -f ${feature} -l ${loss} --num_experts ${experts}
            fi
        done
    done
done
