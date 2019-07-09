""" Utility functions to work with video and frame data. """

from tensorflow.python.framework.ops import Tensor
from tensorflow.python.data import TFRecordDataset
import tensorflow as tf


class Youtube8mData:
    """Youtube 8m tfrecord file reader class.

    Parameters
    ----------
    records: list of str
       A list of strings of TFRecord file names
    feature: str
       A feature selection for the data. Options are 'rgb', 'audio', all
    merged: bool
        If true, return merged rgb and audio features. Valid if feature is all.
        Otherwise ignored.
    parser_fn: str
       Select sequence or example data. Options are 'example' and 'sequence'
    train: bool
       If True returns features, labels. Otherwise returns just features(Useful
       for evalutation mode)
    repeats: int, None
       How many times to iterate over the data. None means repeat indefinitely
    cores: int
       Number of parallel jobs for mapping function.
    batch_size: int
       Number of batch size
    prefetch: bool
       Prefetch is used to speed up the parallelism when using with GPU.
    one_hot: bool
        Label return type. If false, sparse labels are returned
    num_classes: int
       Number of output classes.
    """

    def __init__(self, records,
                 feature,
                 merged=False,
                 parser_fn='example',
                 train=True,
                 repeats=1000,
                 cores=4,
                 batch_size=32,
                 prefetch=True,
                 one_hot=True,
                 num_classes=3862):

        self._records = records
        self._feature = feature
        self._merged = merged
        self._parser_fn = parser_fn
        self._train = train
        self._repeats = repeats
        self._cores = cores
        self._batch_size = batch_size
        self._prefetch = prefetch
        self._num_classes = num_classes
        self._one_hot = one_hot

        if self._feature not in ['rgb', 'audio', 'all']:
            raise ValueError('Please enter a valid selection to extract from TFRecord files. '
                             'Options are rgb, audio and all.')

        if self._parser_fn not in ['example', 'sequence']:
            raise ValueError('Please enter a valid parser function. '
                             'Options are example and sequence.')

        # Hidden parameters from user
        self._shape = {'rgb': 1024, 'audio': 128}
        self._max_frames = 300

    @staticmethod
    def _resize_axis(tensor, axis, new_size, fill_value=0):
        """Truncates or pads a tensor to new_size on on a given axis.
        Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
        size increases, the padding will be performed at the end, using fill_value.

        Parameters
        ----------
        tensor: Tensor
            The tensor to be resized.
        axis: int
            An integer representing the dimension to be sliced.
        new_size: int, Tensor
            An integer or 0d tensor representing the new value for tensor.shape[axis].
        fill_value: int
            Value to use to fill any new entries in the tensor.
            Will be cast to the type of tensor.

        Returns
        -------
        resized: Tensor
            The resized tensor.
        """

        tensor = tf.convert_to_tensor(tensor)
        shape = tf.unstack(tf.shape(tensor))

        pad_shape = shape[:]
        pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

        shape[axis] = tf.minimum(shape[axis], new_size)
        shape = tf.stack(shape)

        resized = tf.concat([
            tf.slice(tensor, tf.zeros_like(shape), shape),
            tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
        ], axis)

        # Update shape.
        new_shape = tensor.get_shape().as_list()  # A copy is being made.
        new_shape[axis] = new_size
        resized.set_shape(new_shape)
        return resized

    def _get_video_matrix(self, features, max_quantized_value=2, min_quantized_value=-2):
        """Decodes features from an input string and dequantizes it.

        Parameters
        ----------
        features: Tensor
            Raw feature values
        max_quantized_value: int
            Maximum of the quantized value.
        min_quantized_value: int
            Minimum of the quantized value.

        Returns
        -------
        resized: Tensor
            Feature_matrix of all frame-features
        """

        # Decode the features into float type and reshape them.
        decoded_features = tf.cast(tf.decode_raw(features, tf.uint8), tf.float32)
        decoded_features = tf.reshape(decoded_features, [-1, self._shape[self._feature]])

        # Dequantize the features from the byte format to the float format
        quantized_range = max_quantized_value - min_quantized_value
        scalar = quantized_range / 255.0
        bias = (quantized_range / 512.0) + min_quantized_value
        feature_matrix = decoded_features * scalar + bias

        # Reshape the features and fill empty frames with 0's
        return self._resize_axis(feature_matrix, 0, self._max_frames)

    def _parser_sequence(self, record):
        """ Parses Sequence data fom TFRecord file to tensors.

        Parameters
        ----------
        record: str
            TFRecord file to parse a single sequence from.

        Returns
        -------
        x or x, y: Tensor or Tensor, Tensor
            Tensor features, labels if train is True, otherwise just features
        """

        context_features = {
            'id': tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.VarLenFeature(tf.int64)
        }
        sequence_features = {
            'rgb': tf.io.FixedLenSequenceFeature([], tf.string),
            'audio': tf.io.FixedLenSequenceFeature([], tf.string)
        }

        context, features = tf.io.parse_single_sequence_example(serialized=record,
                                                                context_features=context_features,
                                                                sequence_features=sequence_features)

        if self._train:
            if self._one_hot:  # Conver to one-hot encoded tensors
                y = tf.sparse_to_dense(tf.sort(context["labels"].values), [self._num_classes], 1)
            else:  # Sparse tensors
                y = tf.sort(context["labels"].values)

            if (self._feature == 'rgb') or (self._feature == 'audio'):
                x = self._get_video_matrix(features[self._feature], )
                return x, y
            else:
                rgb = self._get_video_matrix(features['rgb'], self._shape['rgb'])
                audio = self._get_video_matrix(features['audio'], self._shape['audio'])
                return {'rgb': rgb, 'audio': audio}, y

    def _parser_example(self, record):
        """ Parses Example data fom TFRecord file to tensors.

        Parameters
        ----------
        record: str
            TFRecord file to parse a single example from.

        Returns
        -------
        x or x, y or dict(x1, x2) ,y: Tensor or Tensor, Tensor, dict(Tensor, Tensor), Tensor
            Tensor features, labels if train is True, otherwise just features
        """

        feature_map = {
            "mean_rgb": tf.io.FixedLenFeature([1024], tf.float32),
            "mean_audio": tf.io.FixedLenFeature([128], tf.float32),
            "labels": tf.io.VarLenFeature(tf.int64)
        }

        parsed = tf.io.parse_single_example(record, feature_map)

        if self._train:
            if self._one_hot: # Conver to one-hot encoded tensors
                y = tf.sparse_to_dense(tf.sort(parsed["labels"].values), [self._num_classes], 1)
            else:   # Sparse tensors
                y = tf.sort(parsed["labels"].values)

            if (self._feature == 'rgb') or (self._feature == 'audio'):
                return parsed['mean_' + self._feature], y
            elif self._merged:   # check if we want to merge features together
                return tf.concat([parsed['mean_rgb'], parsed['mean_audio']], axis=0), y
            else:   # Return separate features
                return {'rgb': parsed['mean_rgb'], 'audio': parsed['mean_audio']}, y

        else:     # Eval or Test mode. Skipping label values
            if (self._feature == 'rgb') or (self._feature == 'audio'):
                return parsed['mean_' + self._feature],
            elif self._merged:   # check if we want to merge features together
                return tf.concat([parsed['mean_rgb'], parsed['mean_audio']], axis=0)
            else:   # Return separate features
                return {'rgb': parsed['mean_rgb'], 'audio': parsed['mean_audio']}

    def get_data(self):
        """ Prepares the ETL for the model. Reads, maps and returns the data.

        Returns
        -------
        x: TFRecordDataset
            A Dataset object from record files.
        """

        dataset = tf.data.TFRecordDataset(self._records)

        if self._parser_fn == 'example':
            func = self._parser_example
        else:
            func = self._parser_sequence

        if self._train:
            dataset = (dataset.shuffle(buffer_size=1000)
                              .repeat(self._repeats)
                              .map(map_func=func, num_parallel_calls=self._cores)
                              .batch(self._batch_size))
            if self._prefetch:
                dataset = dataset.prefetch(1)
        else:
            dataset = (dataset.map(map_func=func, num_parallel_calls=self._cores)
                              .batch(self._batch_size))

        return dataset
