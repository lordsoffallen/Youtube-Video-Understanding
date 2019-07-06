from youtube8m.train import train_model, create_model
from youtube8m.train_utils import find_steps_per_epoch
from youtube8m.data import Youtube8mData
from youtube8m.utils import dump_results
from glob import glob
import argparse


parser = argparse.ArgumentParser(description='A Script to use when training a model. The model evaluates'
                                             'top k predictions when training but after training ends'
                                             'you should run the eval_inference script to see the metric'
                                             'values for your validation data.')

parser.add_argument('--train_records', type=str, default='video/train/',
                    help='Path to train record files folder. Add backslash at the end!')

parser.add_argument('--val_records', type=str, default='video/val/',
                    help='Path to val record files folder.Add backslash at the end!')

parser.add_argument('--model', type=str, default='logistic',
                    choices=['mlp', 'cnn', 'gru', 'lstm', 'logistic', 'resnet', 'moe'],
                    help='Select a model to train')

parser.add_argument('--loss_fn', type=str, default='huber', choices=['huber', 'binary', 'hamming'],
                    help='Loss function to use in model. Huber and binary crossentropy are performing well')

parser.add_argument('--units', type=str, default=None,
                    help='Unit size parameter for the model. E.g : "128, 256, 512"')

parser.add_argument('--feature', type=str, default='rgb', choices=['audio', 'rgb', 'all'],
                    help='Which feature to use when training the model')

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--cores', type=int, default=4,
                    help='Number of CPU cores available to use')

parser.add_argument('--steps_per_epoch', type=int, default=0,
                    help='Steps per epoch is number of samples / batch size. Default value'
                         'is 0 which invokes a function to find the value. If it is -1'
                         'then it will stay as None for keras to take care of the value.'
                         'This approach may not work in older versions of TensorFlow.')

parser.add_argument('--validation_steps', type=int, default=0,
                    help='Same as steps_per_epoch but for validation data')

parser.add_argument('--checkpoint',  action='store_true',
                    help='Indicates whether to save model or not at each epoch')

parser.add_argument('--tensorboard', action='store_true',
                    help='Dump tensorboard log files  to current directory')

parser.add_argument('--gpu', action='store_true',
                    help='Use NVIDIA GPU if available')

args = parser.parse_args()


if __name__ == '__main__':
    # Read record files from the data directory
    train_records = glob(args.train_records + 'train*.tfrecord')
    val_records = glob(args.val_records + 'val*.tfrecord')
    val_records = val_records[:10]

    # Init steps_per_epoch and validation_steps variables
    if args.steps_per_epoch == 0:
        steps_per_epoch = find_steps_per_epoch(train_records, args.batch_size)
    elif args.steps_per_epoch < 0:
        steps_per_epoch = None
    else:   # If known beforehand, just change it to known int value.
        steps_per_epoch = args.steps_per_epoch

    if args.validation_steps == 0:
        validation_steps = find_steps_per_epoch(val_records, args.batch_size)
    elif args.validation_steps < 0:
        validation_steps = None
    else:   # If known beforehand, just change it to known int value.
        validation_steps = args.validation_steps

    # Working on video data therefore using example parser.
    kwargs = dict(parser_fn='example', train=True, repeats=None,
                  cores=args.cores, batch_size=args.batch_size,
                  prefetch=True if args.gpu is True else False,
                  num_classes=3862, merged=True, one_hot=True)

    # Construct the train and validation datasets
    train_data = Youtube8mData(train_records, args.feature, **kwargs).get_data()
    val_data = Youtube8mData(val_records, args.feature, **kwargs).get_data()

    # Parse model argument
    if args.model is 'moe':
        # TODO Cannot checkpoint moe model because of the lambda layers.
        args.checkpoint = False

    units = None
    if args.units:
        units = list(map(int, args.units.split(',')))
        if len(units) == 1:   # units contains a single element
            units = units[0]

    # Parse input shape
    if args.feature is 'rgb':
        input_shape = (1024,)
        fname = args.model + '_rgb.pkl'
    elif args.feature is 'audio':
        input_shape = (128,)
        fname = args.model + '_audio.pkl'
    else:   # TODO : Combined model
        input_shape = (1024+128,)
        fname = args.model + '_all.pkl'

    # Create the model
    if (args.model is 'lstm') or (args.model is 'gru'):
        kwargs = dict(summary=True, gpu=args.gpu)
    else:
        kwargs = dict(summary=True)

    model = create_model(units, choice=args.model, loss_fn=args.loss_fn,
                         input_shape=input_shape, **kwargs)

    # Start training the model
    history = train_model(model, train_data, val_data,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          tensorboard=args.tensorboard,
                          checkpoint=args.checkpoint,
                          model_name=args.model)

    model_history, param_history = {}, {}
    model_history[args.model.upper()] = history
    param_history[args.model.upper() + '_PARAMS'] = model.count_params()

    dump_results(model_history, param_history, fname='pickles/'+fname)
