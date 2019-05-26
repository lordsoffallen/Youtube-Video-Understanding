from src import train_model, dump_results
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_records', type=str, default='video/',
                    help='Path to train record files folder. Add backslash at the end!')
parser.add_argument('--val_records', type=str, default='video/',
                    help='Path to val record files folder.Add backslash at the end!')
parser.add_argument('--model', type=str, default='dnn', help='Model type. Options are cnn and dnn')
parser.add_argument('--units', type=int, default=256, help='Unit size parameter for the model.')
parser.add_argument('--feature', type=str, default='rgb', choices=['audio', 'rgb'],
                    help='Which feature to use when training the model')
parser.add_argument('--steps_per_epoch', type=int, default=60,
                    help='Steps per epoch is number of samples / batch size')
parser.add_argument('--validation_steps', type=int, default=60,
                    help='Validation steps per epoch is number of samples / batch size')
parser.add_argument('--cores', type=int, default=2,
                    help='Number of CPU cores available to use')
parser.add_argument('--tensorboard', type=bool, default=False,
                    help='Dump tensorboard log files  to current directory')
parser.add_argument('--gpu', type=bool, default=True,
                    help='Use NVIDIA GPU if available')
parser.add_argument('--rgb_file_name', type=str, default='rgb_results.pickle',
                    help='File name of the rgb results.')
parser.add_argument('--audio_file_name', type=str, default='audio_results.pickle',
                    help='File name of the audio results.')
args = parser.parse_args()


if __name__ == '__main__':
    train_records = glob(args.train_records + 'train*.tfrecord')
    val_records = glob(args.train_records + 'val*.tfrecord')

    if args.feature == 'rgb':

        model, params = train_model(train_records=train_records,
                                    val_records=val_records,
                                    sel=args.feature,
                                    units=args.units,
                                    steps_per_epoch=args.steps_per_epoch,
                                    validation_steps=args.validation_steps,
                                    input_shape=(1024, ),
                                    cores=args.cores,
                                    batch_size=args.batch_size,
                                    choice=args.model,
                                    tensorboard=args.tensorboard,
                                    gpu=args.gpu)

        fname = args.model + 'rgb.pickle'
        dump_results(model, params, fname=fname)
    else:
        model, params = train_model(train_records=train_records,
                                    val_records=val_records,
                                    sel=args.feature,
                                    units=args.units,
                                    steps_per_epoch=args.steps_per_epoch,
                                    validation_steps=args.validation_steps,
                                    input_shape=(128,),
                                    cores=args.cores,
                                    batch_size=args.batch_size,
                                    choice=args.model,
                                    tensorboard=args.tensorboard,
                                    gpu=args.gpu)

        fname = args.model + 'audio.pickle'
        dump_results(model, params, fname=fname)
