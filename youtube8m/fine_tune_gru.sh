#!/usr/bin/env bash


for feature in rgb audio all ; do
    echo
    echo "FINE TUNING LSTM ON $feature FEATURES...."
    echo
    for i in 512 1024 2048; do
        for loss in binary huber ; do
            echo "Training model with $i unit parameters....."
            python train_inference.py -m gru -u "$i" -f ${feature} -l ${loss} --gpu

            echo "Training model with $i, $i unit parameters....."
            python train_inference.py -m gru -u "$i, $i" -f ${feature} -l ${loss} --gpu

            echo "Training model with $i, $i, $i unit parameters....."
            python train_inference.py -m gru -u "$i, $i, $i" -f ${feature} -l ${loss} --gpu

            for norm in batch_normalization dropout ; do
                echo "Training model with $i unit parameters ($loss) and ($norm)....."
                python train_inference.py -m gru -u "$i" -f ${feature} -l ${loss} --${norm} --gpu

                echo "Training model with $i, $i unit parameters ($loss) and ($norm)....."
                python train_inference.py -m gru -u "$i, $i" -f ${feature} -l ${loss} --${norm} --gpu

                echo "Training model with $i, $i, $i unit parameters ($loss) and ($norm)....."
                python train_inference.py -m gru -u "$i, $i, $i" -f ${feature} -l ${loss} --${norm} --gpu
            done
        done
    done
done