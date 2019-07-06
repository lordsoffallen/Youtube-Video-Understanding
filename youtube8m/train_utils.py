from tensorflow.python.keras.metrics import MeanMetricWrapper
import tensorflow as tf


def hamming_loss(y_true, y_pred, mode='multilabel'):
    if mode not in ['multiclass', 'multilabel']:
        raise TypeError('mode must be: [None, multilabel])')

    if mode == 'multiclass':
        nonzero = tf.cast(tf.count_nonzero(y_true * y_pred, axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(tf.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


class HammingLoss(MeanMetricWrapper):
    def __init__(self, name='hamming_loss', dtype=None, mode='multilabel'):
        super(HammingLoss, self).__init__(hamming_loss, name, dtype=dtype, mode=mode)


def find_steps_per_epoch(files_list, batch_size):
    """ Finds the steps_per_epoch value from Dataset class.

    Parameters
    ----------
    files_list: list of str
        List of file names probably returned from glob. Files should have .record extension
    batch_size: int
        Number of batch size

    Returns
    -------
    steps: int
        Returns computed steps_per_epoch value
    """

    print('Computing data size. This may take a while...')
    data_len = sum([len([1 for _ in tf.io.tf_record_iterator(f)]) for f in files_list])
    print('Steps_per_epoch should be length of data / batch size.')
    print('{} / {} =~ {}'.format(data_len, batch_size, round(data_len / batch_size)))
    return round(data_len / batch_size)
