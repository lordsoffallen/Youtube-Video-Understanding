from tensorflow.python.keras.models import load_model
from youtube8m import eval_model, Youtube8mData
from youtube8m.models import MixtureOfExpertsModel
from glob import glob
import argparse


parser = argparse.ArgumentParser(description='A Script to use when evaluating a model performance on '
                                             'different metrics.')

parser.add_argument('-d', '--val_records', type=str, default='video/val/',
                    help='Path to validation record files folder. (Add backslash at the end!) '
                         'Default location is video/val/')

parser.add_argument('-m', '--model_dir', type=str, required=True, help='Keras Model h5 file directory.')

parser.add_argument('--model_weights', type=str, default=None,
                    help='Keras Model weights file directory. This is used with '
                         'MoE model which cant be saved due to Lambda Layers.')

parser.add_argument('-f', '--feature', type=str, default='rgb', choices=['audio', 'rgb', 'all'],
                    help='Which feature was used when training the model. Default is rgb')

parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('-c', '--cores', type=int, default=4,
                    help='Number of CPU cores available to use. Default is 4')

args = parser.parse_args()


if __name__ == '__main__':
    # Read record files from the data directory
    val_records = glob(args.val_records + 'val*.tfrecord')
    val_records = val_records[:30]

    # Working on video data therefore using example parser. Train true because we need labels
    kwargs = dict(parser_fn='example', train=True, repeats=1,
                  cores=args.cores, batch_size=args.batch_size,
                  prefetch=False, num_classes=3862, merged=True, one_hot=True)

    # Construct validation dataset
    val_data = Youtube8mData(val_records, args.feature, **kwargs).get_data()

    # Load model
    if args.model_weights:
        model = MixtureOfExpertsModel().create()
        model.load_weights(args.model_weights)
    else:
        model = load_model(args.model_dir, compile=False)

    # Eval results
    eval_model(model, val_data)
