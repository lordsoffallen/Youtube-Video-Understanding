from src import fine_tune_model, dump_results
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_records', type=str, default='video/',
                    help='Path to train record files folder. Add backslash at the end!')
parser.add_argument('--val_records', type=str, default='video/',
                    help='Path to val record files folder.Add backslash at the end!')
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
                  gpu=True, repeats=None, fname='rgb_results.pickle'):
    """ Fine tunes different models on rgb values and dumps the results as a pickle
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
    fname: str
        File name of the file to save.
    """

    dnn_rgb, dnn_rgb_params = fine_tune_model(train_records=train_records,
                                              val_records=val_records,
                                              sel='rgb',
                                              repeats=repeats,
                                              cores=cores,
                                              choice='dnn',
                                              tensorboard=tensorboard)

    cnn_rgb, cnn_rgb_params = fine_tune_model(train_records=train_records,
                                              val_records=val_records,
                                              sel='rgb',
                                              repeats=repeats,
                                              cores=cores,
                                              choice='cnn',
                                              tensorboard=tensorboard,
                                              units=units,
                                              strides=2)

    lstm_rgb, lstm_rgb_params = fine_tune_model(train_records=train_records,
                                                val_records=val_records,
                                                sel='rgb',
                                                repeats=repeats,
                                                cores=cores,
                                                choice='lstm',
                                                tensorboard=tensorboard,
                                                units=units,
                                                gpu=gpu)

    gru_rgb, gru_rgb_params = fine_tune_model(train_records=train_records,
                                              val_records=val_records,
                                              sel='rgb',
                                              repeats=repeats,
                                              cores=cores,
                                              choice='gru',
                                              tensorboard=tensorboard,
                                              units=units,
                                              gpu=gpu)

    rgb_models = {**dnn_rgb, **cnn_rgb, **lstm_rgb, **gru_rgb}
    rgb_params = {**dnn_rgb_params, **cnn_rgb_params, **lstm_rgb_params, **gru_rgb_params}
    dump_results(rgb_models, rgb_params, fname=fname)


def fine_tune_audio(train_records, val_records, units, cores=2, tensorboard=False,
                    gpu=True, repeats=None, fname='audio_results.pickle'):
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
    fname: str
        File name of the file to save.
    """

    dnn_audio, dnn_audio_params = fine_tune_model(train_records=train_records,
                                                  val_records=val_records,
                                                  sel='audio',
                                                  input_shape=(128,),
                                                  repeats=repeats,
                                                  cores=cores,
                                                  choice='dnn',
                                                  tensorboard=tensorboard)

    cnn_audio, cnn_audio_params = fine_tune_model(train_records=train_records,
                                                  val_records=val_records,
                                                  sel='audio',
                                                  input_shape=(128,),
                                                  repeats=repeats,
                                                  cores=cores,
                                                  choice='cnn',
                                                  tensorboard=tensorboard,
                                                  units=units,
                                                  strides=2)

    lstm_audio, lstm_audio_params = fine_tune_model(train_records=train_records,
                                                    val_records=val_records,
                                                    sel='audio',
                                                    input_shape=(128,),
                                                    repeats=repeats,
                                                    cores=cores,
                                                    choice='lstm',
                                                    tensorboard=tensorboard,
                                                    units=units,
                                                    gpu=gpu)

    gru_audio, gru_audio_params = fine_tune_model(train_records=train_records,
                                                  val_records=val_records,
                                                  sel='audio',
                                                  input_shape=(128,),
                                                  repeats=repeats,
                                                  cores=cores,
                                                  choice='gru',
                                                  tensorboard=tensorboard,
                                                  units=units,
                                                  gpu=gpu)

    audio_models = {**dnn_audio, **cnn_audio, **lstm_audio, **gru_audio}
    audio_params = {**dnn_audio_params, **cnn_audio_params, **lstm_audio_params, **gru_audio_params}
    dump_results(audio_models, audio_params, fname=fname)


if __name__ == '__main__':

    train = glob(args.train_records + 'train*.tfrecord')
    val = glob(args.train_records + 'val*.tfrecord')

    units_cnn_lstm_gru = [[128, 128, 128], [128, 256, 256], [128, 256, 512],
                          [256, 256, 256], [256, 256, 512], [256, 512, 512]]

    fine_tune_rgb(train_records=train, val_records=val, units=units_cnn_lstm_gru,
                  cores=args.cores, tensorboard=args.tensorboard, gpu=args.gpu,
                  repeats=None, fname=args.rgb_file_name)

    fine_tune_audio(train_records=train, val_records=val, units=units_cnn_lstm_gru,
                    cores=args.cores, tensorboard=args.tensorboard, gpu=args.gpu,
                    repeats=None, fname=args.audio_file_name)



