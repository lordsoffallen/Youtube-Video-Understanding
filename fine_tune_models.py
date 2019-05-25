from src import fine_tune_model, dump_results
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_records', type=str, default='video/',
                    help='Path to train record files folder. Add backslash at the end!')
parser.add_argument('--val_records', type=str, default='video/',
                    help='Path to val record files folder.Add backslash at the end!')
parser.add_argument('--steps_per_epoch', type=int, default=60,
                    help='Steps per epoch is number of samples / batch size')
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


def fine_tune_rgb(train_records, val_records, units, cores=2, tensorboard=False,
                  gpu=True, repeats=None, steps_per_epoch=60,
                  fname='rgb_results.pickle'):
    """ Fine tunes different models on rgb values and dumps the results as a pickle
    object to read later.

    Parameters
    ----------
    train_records: str
        A list of strings of TFRecord train file names
    val_records: str
        A list of strings of TFRecord validation file names
    units: dict
        A dict contains number of units in each layer. Dict keys should be
        in the form of (rgb or audio)_(model name)_units
    cores: int
        Number of parallel jobs for mapping function.
    tensorboard: bool
        Whether to output tensorboard logs. Default is false
    gpu: bool
        If GPU is enabled, then set this to true to use cuDNNLSTM
        which is much faster.
    repeats: int
        How many times to iterate over the data
    steps_per_epoch: int
        Number of steps required to complete one training part.
    fname: str
        File name of the file to save.
    """

    dnn_rgb, dnn_rgb_params = fine_tune_model(train_records=train_records,
                                              val_records=val_records,
                                              sel='rgb',
                                              steps_per_epoch=steps_per_epoch,
                                              repeats=repeats,
                                              cores=cores,
                                              units=units['rgb_dnn_units'],
                                              choice='dnn',
                                              tensorboard=tensorboard)

    cnn_rgb, cnn_rgb_params = fine_tune_model(train_records=train_records,
                                              val_records=val_records,
                                              sel='rgb',
                                              steps_per_epoch=steps_per_epoch,
                                              repeats=repeats,
                                              cores=cores,
                                              choice='cnn',
                                              tensorboard=tensorboard,
                                              units=units['rgb_cnn_units'],
                                              strides=2)

    lstm_rgb, lstm_rgb_params = fine_tune_model(train_records=train_records,
                                                val_records=val_records,
                                                sel='rgb',
                                                steps_per_epoch=steps_per_epoch,
                                                repeats=repeats,
                                                cores=cores,
                                                choice='lstm',
                                                tensorboard=tensorboard,
                                                units=units['rgb_lstm_units'],
                                                gpu=gpu)

    gru_rgb, gru_rgb_params = fine_tune_model(train_records=train_records,
                                              val_records=val_records,
                                              sel='rgb',
                                              steps_per_epoch=steps_per_epoch,
                                              repeats=repeats,
                                              cores=cores,
                                              choice='gru',
                                              tensorboard=tensorboard,
                                              units=units['rgb_gru_units'],
                                              gpu=gpu)

    rgb_models = {**dnn_rgb, **cnn_rgb, **lstm_rgb, **gru_rgb}
    rgb_params = {**dnn_rgb_params, **cnn_rgb_params, **lstm_rgb_params, **gru_rgb_params}
    dump_results(rgb_models, rgb_params, fname=fname)


def fine_tune_audio(train_records, val_records, units, cores=2, tensorboard=False,
                    gpu=True, repeats=None, steps_per_epoch=60,
                    fname='audio_results.pickle'):
    """ Fine tunes different models on audio values and dumps the results as a pickle
    object to read later.

    Parameters
    ----------
    train_records: str
        A list of strings of TFRecord train file names
    val_records: str
        A list of strings of TFRecord validation file names
    units: int, list
        Number of units in each layer
    cores: int
        Number of parallel jobs for mapping function.
    tensorboard: bool
        Whether to output tensorboard logs. Default is false
    gpu: bool
        If GPU is enabled, then set this to true to use cuDNNLSTM
        which is much faster.
    repeats: int
        How many times to iterate over the data
    steps_per_epoch: int
        Number of steps required to complete one training part.
    fname: str
        File name of the file to save.
    """

    dnn_audio, dnn_audio_params = fine_tune_model(train_records=train_records,
                                                  val_records=val_records,
                                                  sel='audio',
                                                  steps_per_epoch=steps_per_epoch,
                                                  input_shape=(128,),
                                                  repeats=repeats,
                                                  cores=cores,
                                                  units=units['audio_dnn_units'],
                                                  choice='dnn',
                                                  tensorboard=tensorboard)

    cnn_audio, cnn_audio_params = fine_tune_model(train_records=train_records,
                                                  val_records=val_records,
                                                  sel='audio',
                                                  steps_per_epoch=steps_per_epoch,
                                                  input_shape=(128,),
                                                  repeats=repeats,
                                                  cores=cores,
                                                  choice='cnn',
                                                  tensorboard=tensorboard,
                                                  units=units['audio_cnn_units'],
                                                  strides=2)

    lstm_audio, lstm_audio_params = fine_tune_model(train_records=train_records,
                                                    val_records=val_records,
                                                    sel='audio',
                                                    steps_per_epoch=steps_per_epoch,
                                                    input_shape=(128,),
                                                    repeats=repeats,
                                                    cores=cores,
                                                    choice='lstm',
                                                    tensorboard=tensorboard,
                                                    units=units['audio_lstm_units'],
                                                    gpu=gpu)

    gru_audio, gru_audio_params = fine_tune_model(train_records=train_records,
                                                  val_records=val_records,
                                                  sel='audio',
                                                  steps_per_epoch=steps_per_epoch,
                                                  input_shape=(128,),
                                                  repeats=repeats,
                                                  cores=cores,
                                                  choice='gru',
                                                  tensorboard=tensorboard,
                                                  units=units['audio_gru_units'],
                                                  gpu=gpu)

    audio_models = {**dnn_audio, **cnn_audio, **lstm_audio, **gru_audio}
    audio_params = {**dnn_audio_params, **cnn_audio_params, **lstm_audio_params, **gru_audio_params}
    dump_results(audio_models, audio_params, fname=fname)


if __name__ == '__main__':

    train = glob(args.train_records + 'train*.tfrecord')
    val = glob(args.train_records + 'val*.tfrecord')

    audio_gru_units = [[128, 128, 128], [128, 256, 256], [128, 256, 512]]
    audio_lstm_units = [[128, 128, 128], [128, 256, 256], [256, 256, 256]]
    audio_cnn_units = [[128, 128, 128], [128, 256, 512]]
    audio_dnn_units = [256, 512, 1024, [256, 512], [512, 1024]]

    rgb_gru_units = [[128, 128, 128], [128, 256, 256], [256, 256, 256]]
    rgb_lstm_units = [[256, 256, 256], [256, 256, 512], [256, 512, 512]]
    rgb_cnn_units = [[128, 128, 128], [256, 512, 512]]
    rgb_dnn_units = [256, 512, 1024, [256, 512], [512, 1024]]

    rgb_units = {
        'rgb_dnn_units': rgb_dnn_units,
        'rgb_cnn_units': rgb_cnn_units,
        'rgb_lstm_units': rgb_lstm_units,
        'rgb_gru_units': rgb_gru_units
    }

    audio_units = {
        'audio_dnn_units': audio_dnn_units,
        'audio_cnn_units': audio_cnn_units,
        'audio_lstm_units': audio_lstm_units,
        'audio_gru_units': audio_gru_units
    }

    fine_tune_rgb(train_records=train, val_records=val, units=rgb_units,
                  cores=args.cores, tensorboard=args.tensorboard, gpu=args.gpu,
                  repeats=None, steps_per_epoch=args.steps_per_epoch, fname=args.rgb_file_name)

    fine_tune_audio(train_records=train, val_records=val, units=audio_units,
                    cores=args.cores, tensorboard=args.tensorboard, gpu=args.gpu,
                    repeats=None, steps_per_epoch=args.steps_per_epoch, fname=args.audio_file_name)



