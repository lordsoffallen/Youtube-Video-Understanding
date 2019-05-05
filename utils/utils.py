""" Utility functions to work with video and frame data. """

import tensorflow as tf


def resize_axis(tensor, axis, new_size, fill_value=0):
    """Truncates or pads a tensor to new_size on on a given axis.
    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.
    
    Args:
        tensor: The tensor to be resized.
        axis: An integer representing the dimension to be sliced.
        new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
        fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.
        
    Returns:
        The resized tensor.
    """

    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size-shape[axis])

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

def get_video_matrix(features, feature_size, max_frames=300, 
                     max_quantized_value=2, min_quantized_value=-2):
    """Decodes features from an input string and dequantizes it.
    
    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.
      
    Returns:
      feature_matrix: matrix of all frame-features
    """
    
    # Decode the features into float type and reshape them.
    decoded_features = tf.cast(tf.decode_raw(features, tf.uint8), tf.float32)
    decoded_features = tf.reshape(decoded_features, [-1, feature_size])
    
    # Get the number of frames
    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    
    # Dequantize the features from the byte format to the float format
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    feature_matrix = decoded_features * scalar + bias 

    # Reshape the features and fill empty frames with 0's
    return resize_axis(feature_matrix, 0, max_frames)

def parser_sequence(record, sel, train=True, num_classes=3862):
    """ Parses Sequence data fom TFRecord file to tensors.

    Args:
        record: TFRecord file to parse a single example from.
        sel: Select which feature to extract. Options are 'rgb', 'audio'
        train: if True returns features, labels. Otherwise returns just features 
        num_classes: Number of output classes. Default values is 3862

    Returns:
        Tensor features, labels if train is True, otherwise just features
    """
    shape = {'rgb': 1024, 'audio': 128}

    if sel not in ['rgb', 'audio']:
        raise('Please enter a valid selection to extract from TFRecord files. Options are \
            rgb and audio.')

    context_features = {
        'id': tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64)
    }
    sequence_features = {
        'rgb': tf.FixedLenSequenceFeature([], tf.string),
        'audio': tf.FixedLenSequenceFeature([], tf.string)
    }
    
    context, features = tf.parse_single_sequence_example(record, 
                              context_features=context_features, 
                              sequence_features=sequence_features)

    # Retrieve y and one-hot encode them.
    x = get_video_matrix(features[sel], shape[sel])
    if train:
        y = tf.sparse_to_dense(tf.contrib.framework.sort(context["labels"].values), [num_classes], 1)
        return x, y
    return x

def parser_example(record, sel, train=True, num_classes=3862):
    """ Parses Sequence data fom TFRecord file to tensors.

    Args:
        record: TFRecord file to parse a single example from.
        sel: Select which feature to extract. Options are 'rgb', 'audio'
        train: if True returns features, labels. Otherwise returns just features 
        num_classes: Number of output classes. Default values is 3862

    Returns:
        Tensor features, labels if train is True, otherwise just features
    """
    
    
    if sel not in ['rgb', 'audio']:
        raise('Please enter a valid selection to extract from TFRecord files. Options are \
            rgb and audio.')

    feature_map = {
        "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
        "mean_audio": tf.FixedLenFeature([128], tf.float32),
        "labels": tf.VarLenFeature(tf.int64)
    }
    
    parsed = tf.parse_single_example(record, feature_map)
    if train:
        y = tf.sparse_to_dense(tf.contrib.framework.sort(parsed["labels"].values), [num_classes], 1)
        return parsed['mean_' + sel], y
    return parsed['mean_' + sel]

def get_data(records, sel, parser='example', train=True, repeats=1000,
             cores=4, batch_size=32, buffer_size=2, num_classes=3862): 
    """ Prepares the ETL for the model. Reads, maps and returns the data.
    
    Args:
        records: A list of strings of TFRecord file names
        sel: A feature selection for the data. Options are 'rgb' and 'audio'
        parser: Select sequence or example data. Options are 'example' and 'sequence'
        train: if True returns features, labels. Otherwise returns just features 
        repeats: How many times to iterate over the data
        cores: Number of parallel jobs for mapping function.
        batch_size: Number of batch size
        buffer_size: A prefetch buffer size to speed up to parallelism
        num_classes: Number of output classes. Default values is 3862
    
    Returns:
        Next Tensor element from the data
    """

    if sel not in ['rgb', 'audio']:
        raise('Please enter a valid selection to extract from TFRecord files. Options are \
            rgb and audio.')

    if parser not in ['example', 'sequence']:
        raise('Please enter a valid selection for parser. Options are \
            example and sequence.')

    dataset = tf.data.TFRecordDataset(records)

    if parser is 'example':
        func = lambda x : parser_example(record=x, sel=sel, train=train, num_classes=num_classes)
    else:
        func = lambda x : parser_sequence(record=x, sel=sel, train=train, num_classes=num_classes)
    
    dataset = (dataset.map(map_func=func, num_parallel_calls=cores)
                      .repeat(repeats)
                      .shuffle(buffer_size=1000)
                      .batch(batch_size)
                      .prefetch(buffer_size))

    one_shot = dataset.make_one_shot_iterator()
    return one_shot.get_next()  

def generator(records, sel, parser='example', train=True, repeats=1000, 
              cores=4, batch_size=32, buffer_size=2, num_classes=3862):
    """ Generates the data from TFRecord files.
    
    Args:
        records: A list of strings of TFRecord file names.
        sel: A feature selection for the data. Options are 'all', 'rgb' and 'audio' 
            When sel is all, then both rgb and audio parser will return but audio data
            will return only features.
        parser: Select sequence or example data. Options are 'example' and 'sequence'
        train: if True returns features, labels. Otherwise returns just features 
        repeats: How many times to iterate over the data.
        cores: Number of parallel jobs for mapping function.
        batch_size: Number of batch size.
        buffer_size: A prefetch buffer size to speed up to parallelism
        num_classes: Number of output classes. Default values is 3862
    
    Returns:
        A Generator to retrieve the data in batches
    """

    if sel not in ['all', 'rgb', 'audio']:
        raise('Please enter a valid selection to extract from TFRecord files. Options are \
              all, rgb and audio.')

    if sel is 'all':
        rgb = get_data(records=records, sel='rgb', parser=parser, train=train,
                       repeats=repeats, cores=cores, batch_size=batch_size, 
                       buffer_size=buffer_size, num_classes=num_classes)

        audio = get_data(records=records, sel='audio', parser=parser, train=False,
                         repeats=repeats, cores=cores, batch_size=batch_size, 
                         buffer_size=buffer_size, num_classes=num_classes)

        sess = tf.Session()
        while True:
            try:
                x1, y = sess.run(rgb)
                x2 = sess.run(audio)
                yield [x1, x2], y
            except tf.errors.OutOfRangeError:
                print("Iterations exhausted")
                sess.close()
                break
    else:
        data = get_data(records=records, sel=sel, parser=parser, train=train,
                        repeats=repeats, cores=cores, batch_size=batch_size, 
                        buffer_size=buffer_size, num_classes=num_classes)

        sess = tf.Session()
        while True:
            try:
                x, y = sess.run(data)
                yield x, y
            except tf.errors.OutOfRangeError:
                print("Iterations exhausted")
                sess.close()
                break

