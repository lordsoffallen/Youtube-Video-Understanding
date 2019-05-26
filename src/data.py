""" Utility functions to work with video and frame data. """

from tensorflow.python import Session
from tensorflow.python.framework.dtypes import uint8, float32, string, int64
from tensorflow.python.framework.ops import Tensor, convert_to_tensor
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.contrib.framework.python.ops.sort_ops import sort
from tensorflow.python.data import TFRecordDataset
from tensorflow.python.ops.gen_math_ops import maximum, minimum
from tensorflow.python.ops.sparse_ops import sparse_to_dense
from tensorflow.python.ops.gen_parsing_ops import decode_raw
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.array_ops import shape as tensor_shape
from tensorflow.python.ops.array_ops import unstack, stack, slice
from tensorflow.python.ops.array_ops import zeros_like, concat, fill
from tensorflow.python.ops.parsing_ops import (FixedLenSequenceFeature,
                                               FixedLenFeature, VarLenFeature,
                                               parse_single_sequence_example,
                                               parse_single_example)


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

    tensor = convert_to_tensor(tensor)
    shape = unstack(tensor_shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = maximum(0, new_size - shape[axis])

    shape[axis] = minimum(shape[axis], new_size)
    shape = stack(shape)

    resized = concat([
        slice(tensor, zeros_like(shape), shape),
        fill(stack(pad_shape), cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized


def _get_video_matrix(features, feature_size, max_frames=300,
                      max_quantized_value=2, min_quantized_value=-2):
    """Decodes features from an input string and dequantizes it.
    
    Parameters
    ----------
    features: Tensor
        Raw feature values
    feature_size: int
        Length of each frame feature vector
    max_frames: int
        Number of frames (rows) in the output feature_matrix
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
    decoded_features = cast(decode_raw(features, uint8), float32)
    decoded_features = reshape(decoded_features, [-1, feature_size])

    # Dequantize the features from the byte format to the float format
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    feature_matrix = decoded_features * scalar + bias

    # Reshape the features and fill empty frames with 0's
    return _resize_axis(feature_matrix, 0, max_frames)


def _parser_sequence(record, sel, train=True, num_classes=3862):
    """ Parses Sequence data fom TFRecord file to tensors.

    Parameters
    ----------
    record: str
        TFRecord file to parse a single example from.
    sel: str
        Select which feature to extract. Options are 'rgb', 'audio'
    train: bool
        If True returns features, labels. Otherwise returns just features
    num_classes: int
        Number of output classes. Default values is 3862

    Returns
    -------
    x or x, y: Tensor or Tensor, Tensor
        Tensor features, labels if train is True, otherwise just features
    """

    shape = {'rgb': 1024, 'audio': 128}

    if sel not in ['rgb', 'audio']:
        raise ('Please enter a valid selection to extract from TFRecord files. Options are \
                rgb and audio.')

    context_features = {
        'id': FixedLenFeature([], string),
        "labels": VarLenFeature(int64)
    }
    sequence_features = {
        'rgb': FixedLenSequenceFeature([], string),
        'audio': FixedLenSequenceFeature([], string)
    }

    context, features = parse_single_sequence_example(record,
                                                      context_features=context_features,
                                                      sequence_features=sequence_features)

    # Retrieve y and one-hot encode them.
    x = _get_video_matrix(features[sel], shape[sel])
    if train:
        y = sparse_to_dense(sort(context["labels"].values), [num_classes], 1)
        return x, y
    return x


def _parser_example(record, sel, train=True, num_classes=3862):
    """ Parses Example data fom TFRecord file to tensors.

    Parameters
    ----------
    record: str
        TFRecord file to parse a single example from.
    sel: str
        Select which feature to extract. Options are 'rgb', 'audio', 'all'
    train: bool
        If True returns features, labels. Otherwise returns just features
    num_classes: int
        Number of output classes. Default values is 3862

    Returns
    -------
    x or x, y or dict(x1, x2) ,y: Tensor or Tensor, Tensor, dict(Tensor, Tensor), Tensor
        Tensor features, labels if train is True, otherwise just features
    """

    if sel not in ['rgb', 'audio', 'all']:
        raise ('Please enter a valid selection to extract from TFRecord files. Options are \
            rgb, audio and all.')

    feature_map = {
        "mean_rgb": FixedLenFeature([1024], float32),
        "mean_audio": FixedLenFeature([128], float32),
        "labels": VarLenFeature(int64)
    }

    parsed = parse_single_example(record, feature_map)
    if train:
        y = sparse_to_dense(sort(parsed["labels"].values), [num_classes], 1)
        if sel is 'all':
            return {'rgb': parsed['mean_rgb', 'audio': parsed['mean_audio']]}, y
        return parsed['mean_' + sel], y
    return parsed['mean_' + sel]


def get_data(records, sel, parser='example', train=True, repeats=1000,
             cores=4, batch_size=32, buffer_size=1, num_classes=3862):
    """ Prepares the ETL for the model. Reads, maps and returns the data.
    
    Parameters
    ----------
    records: str
        A list of strings of TFRecord file names
    sel: str
        A feature selection for the data. Options are 'rgb', 'audio', all
    parser: str
        Select sequence or example data. Options are 'example' and 'sequence'
    train: bool
        If True returns features, labels. Otherwise returns just features
    repeats: int
        How many times to iterate over the data
    cores: int
        Number of parallel jobs for mapping function.
    batch_size: int
        Number of batch size
    buffer_size: int
        A prefetch buffer size to speed up to parallelism
    num_classes: int
        Number of output classes. Default values is 3862
    
    Returns
    -------
    x: TFRecordDataset
        A Dataset object from record files.
    """

    if sel not in ['rgb', 'audio', 'all']:
        raise ('Please enter a valid selection to extract from TFRecord files. Options are \
                rgb, audio and all.')

    if parser not in ['example', 'sequence']:
        raise ('Please enter a valid selection for parser. Options are \
                example and sequence.')

    dataset = TFRecordDataset(records)

    if parser is 'example':
        func = lambda x: _parser_example(record=x, sel=sel, train=train, num_classes=num_classes)
    else:
        func = lambda x: _parser_sequence(record=x, sel=sel, train=train, num_classes=num_classes)

    if train:
        dataset = (dataset.shuffle(buffer_size=1000)
                          .repeat(repeats)
                          .map(map_func=func, num_parallel_calls=cores)
                          .batch(batch_size)
                          .prefetch(buffer_size))
    else:
        dataset = (dataset.map(map_func=func, num_parallel_calls=cores)
                          .batch(batch_size)
                          .prefetch(buffer_size))

    return dataset


def _generator(records, sel, parser='example', train=True, repeats=1000,
              cores=4, batch_size=32, buffer_size=1, num_classes=3862):
    """ Generates the data from TFRecord files. This is an old method
    to use. For now, use get_data function.
    
    Parameters
    ----------
    records: str
        A list of strings of TFRecord file names.
    sel: str
        A feature selection for the data. Options are 'all', 'rgb' and 'audio'
        When sel is all, then both rgb and audio parser will return but audio data
        will return only features.
    parser: str
        Select sequence or example data. Options are 'example' and 'sequence'
    train: bool
        If True returns features, labels. Otherwise returns just features
    repeats: int
        How many times to iterate over the data.
    cores: int
        Number of parallel jobs for mapping function.
    batch_size: int
        Number of batch size.
    buffer_size: int
        A prefetch buffer size to speed up to parallelism
    num_classes: int
        Number of output classes. Default values is 3862
    
    Returns
    -------
    [x1, x2], y or x, y:
        A Generator to retrieve the data in batches. Depending on the sel
        parameter different outputs will return.
    """

    if sel not in ['all', 'rgb', 'audio']:
        raise ('Please enter a valid selection to extract from TFRecord files. Options are \
                all, rgb and audio.')

    if sel is 'all':
        rgb = get_data(records=records, sel='rgb', parser=parser, train=train,
                       repeats=repeats, cores=cores, batch_size=batch_size,
                       buffer_size=buffer_size, num_classes=num_classes)

        audio = get_data(records=records, sel='audio', parser=parser, train=False,
                         repeats=repeats, cores=cores, batch_size=batch_size,
                         buffer_size=buffer_size, num_classes=num_classes)

        sess = Session()
        while True:
            try:
                x1, y = sess.run(rgb.get_next())
                x2 = sess.run(audio.get_next())
                yield [x1, x2], y
            except OutOfRangeError:
                print("Iterations exhausted")
                sess.close()
                break
    else:
        data = get_data(records=records, sel=sel, parser=parser, train=train,
                        repeats=repeats, cores=cores, batch_size=batch_size,
                        buffer_size=buffer_size, num_classes=num_classes)

        sess = Session()
        while True:
            try:
                x, y = sess.run(data.get_next())
                yield x, y
            except OutOfRangeError:
                print("Iterations exhausted")
                sess.close()
                break