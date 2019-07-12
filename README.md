# Youtube Video Understanding 

This repository contains files related to YouTube 8m dataset. 


## Files

- `Youtube Data Analytics.ipynb` - A notebook for analyzing the files contents and discovering
data distributions. This should be the starting point of understanding the data.

- `Youtube8m_Script.ipynb` - This notebook is used on [Google's colab](https://colab.research.google.com) 
to find tune models while training on GPU. 

- `train_inference.py` - This script is used to train data with different models. 
Run `python train_inference.py --help`  to see parameters to pass on

```
usage: train_inference.py [-h] [--train_records TRAIN_RECORDS]
                          [--val_records VAL_RECORDS]
                          [-m {mlp,cnn,gru,lstm,logistic,resnet,moe}]
                          [-l {huber,binary,hamming}] [-o {adam,sgd}]
                          [-u UNITS] [-f {audio,rgb,all}] [-b BATCH_SIZE]
                          [-c CORES] [--steps_per_epoch STEPS_PER_EPOCH]
                          [--validation_steps VALIDATION_STEPS]
                          [--num_experts NUM_EXPERTS] [--batch_normalization]
                          [--dropout] [--checkpoint] [--tensorboard] [--gpu]

A Script to use when training a model. The model evaluatestop 5 predictions
when training but after training endsyou should run the eval_inference script
to see the metricvalues for your validation data.

optional arguments:
  -h, --help            show this help message and exit
  --train_records TRAIN_RECORDS
                        Path to train record files folder. (Add backslash at
                        the end!) Default location is video/train/
  --val_records VAL_RECORDS
                        Path to val record files folder. (Add backslash at the
                        end!) Default location is video/val/
  -m {mlp,cnn,gru,lstm,logistic,resnet,moe}, --model {mlp,cnn,gru,lstm,logistic,resnet,moe}
                        Select a model to train. Default model is logistic.
  -l {huber,binary,hamming}, --loss_fn {huber,binary,hamming}
                        Loss function to use in model. Huber and binary
                        crossentropy are performing well. Default is huber
                        function
  -o {adam,sgd}, --optimizer {adam,sgd}
                        Optimizer to use in model. Default is adam optimizer.
  -u UNITS, --units UNITS
                        Unit size parameter for the model. If multiple, then
                        pass it in string format e.g: "128, 256, 512" or for
                        single units just a 512. Defaul is None
  -f {audio,rgb,all}, --feature {audio,rgb,all}
                        Which feature to use when training the model. Default
                        is rgb.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Default batch size is 32
  -c CORES, --cores CORES
                        Number of CPU cores available to use. Default is 2
  --steps_per_epoch STEPS_PER_EPOCH
                        Steps per epoch is number of samples / batch size.
                        Default valueis 0 which invokes a function to find the
                        value. If it is -1then it will stay as None for keras
                        to take care of the value.This approach may not work
                        in older versions of TensorFlow.
  --validation_steps VALIDATION_STEPS
                        Same as steps_per_epoch but for validation data
  --num_experts NUM_EXPERTS
                        Number of experts parameter value to pass MoE model
                        when creating
  --batch_normalization
                        Indicates whether to use batch normalization in the
                        model
  --dropout             Indicates whether to use dropout in the model
  --checkpoint          Indicates whether to save model or not at each epoch
  --tensorboard         Dump tensorboard log files to current directory
  --gpu                 Use NVIDIA GPU if available. This is specific to RNN
                        models. For other models GPU will be automatically
                        detected.
```

- `eval_inference.py` - This script is used to evaluate the mode when training part
is complete. Run `python eval_inference.py --help` to see parameters to pass on

```
usage: eval_inference.py [-h] [-d VAL_RECORDS] -m MODEL_DIR
                         [--model_weights MODEL_WEIGHTS] [-f {audio,rgb,all}]
                         [-b BATCH_SIZE] [-c CORES]

A Script to use when evaluating a model performance on different metrics.

optional arguments:
  -h, --help            show this help message and exit
  -d VAL_RECORDS, --val_records VAL_RECORDS
                        Path to validation record files folder. (Add backslash
                        at the end!) Default location is video/val/
  -m MODEL_DIR, --model_dir MODEL_DIR
                        Keras Model h5 file directory.
  --model_weights MODEL_WEIGHTS
                        Keras Model weights file directory. This is used with
                        MoE model which cant be saved due to Lambda Layers.
  -f {audio,rgb,all}, --feature {audio,rgb,all}
                        Which feature was used when training the model.
                        Default is rgb
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -c CORES, --cores CORES
                        Number of CPU cores available to use. Default is 4
```

- `youtube8m/data.py` - It contains Youtube8MData class to read data from *.tfrecord files.
This function has a `get_data()` function which creates a tf.data.Dataset class for efficient
reading of data.

- `youtube8m/eval.py` - It contains evaluation function to test model.

- `youtube8m/metric_utils.py` - It contains utility classes for metrics. 

- `youtube8m/metrics.py` - It contains the EvaluationMetrics class for calculating the 
metrics for given model and data.

- `youtube8m/models.py` - It contains model classes. When creating a new model, one should
extend the BaseModel class. 
 
- `youtube8m/train_utils.py` - It contains utility functions for training.

- `youtube8m/train.py` - It contains functions for training. Mainly `create_model()` and
`train_model()`.

- `youtube8m/utils.py` - It contains utility functions for model output plotting and model
history dumping as a pickle file.

- `youtube8m/fine_tune_*.sh` - Bash scripts contains commands to train model on different
parameters. It is used to find the best hyper parameters of the model.




# Paper Links

[Google Original Paper](https://arxiv.org/pdf/1609.08675.pdf)  
[Standford Paper 1](http://cs231n.stanford.edu/reports/2017/pdfs/702.pdf)  
[Standford Paper 2](http://cs231n.stanford.edu/reports/2017/pdfs/711.pdf)  
[Standford Paper 3](http://cs231n.stanford.edu/reports/2017/pdfs/705.pdf)  
