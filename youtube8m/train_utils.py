from tensorflow.python.keras.metrics import MeanMetricWrapper
import tensorflow as tf


# TODO Something happened with hamming. It throws an error now.
def hamming_loss(y_true, y_pred):
    nonzero = tf.cast(tf.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
    return nonzero / y_true.get_shape()[-1]


class HammingLoss(MeanMetricWrapper):
    def __init__(self, name='hamming_loss', dtype=None):
        super(HammingLoss, self).__init__(hamming_loss, name, dtype=dtype)


def hit_at_one(y_true, y_pred, batch_size=32):
    top_prediction = tf.math.argmax(y_pred, axis=1, output_type=tf.int32)
    hits = tf.gather_nd(y_true, tf.stack([tf.range(batch_size), top_prediction], axis=1))
    return tf.reduce_mean(hits)


def hit_at_n(y_true, y_pred, batch_size=32, n=3):
    _, top_prediction = tf.math.top_k(y_pred, k=n)
    top_prediction = tf.reshape(top_prediction, [batch_size*n])
    repeated_batch = tf.keras.backend.repeat_elements(tf.range(batch_size), n, axis=0)
    hits = tf.gather_nd(y_true, tf.stack([repeated_batch, top_prediction], axis=1))
    return tf.reduce_mean(tf.cast(hits, tf.float32))


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
