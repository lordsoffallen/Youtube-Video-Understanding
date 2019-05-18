from .metrics import top5_acc, top1_acc
from .data import get_data
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, LSTM, GRU
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Reshape
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.backend import clear_session


def dnn_model(units, input_shape=(1024,), num_classes=3862):
    """ Create a DNN model.

    Parameters
    ----------
    units: int, list
        Number of units in dense layer. If given an iterable,
        then will create different layers for each units.
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.

    Returns
    -------
    model: Model
        A DNN model
    """

    model = Sequential()
    if isinstance(units, int):
        model.add(Dense(units, activation='relu', input_shape=input_shape))
    else:  # Retrieve first unit and add input shape
        model.add(Dense(units.pop(0), activation='relu', input_shape=input_shape))
        for unit in units:
            model.add(Dense(unit, activation='relu'))

    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def cnn_model(units, pooling=False, input_shape=(1024,), num_classes=3862):
    """ Create a CNN model. Default kernel size is (3, 3)

    Parameters
    ----------
    units: int, list
        Number of units in convolution layer. If given an iterable,
        then will create different layers for each units.
    pooling: bool
        Whether to add pooling after conv layer or not. Default is False
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.

    Returns
    -------
    model: Model
        A CNN model
    """

    model = Sequential()
    model.add(Reshape((1, input_shape[0]), input_shape=input_shape))
    if isinstance(units, int):
        model.add(Conv1D(units, 3, padding='same', activation='relu'))
        if pooling:
            model.add(MaxPooling1D())
    else:
        for unit in units:
            model.add(Conv1D(unit, 3, padding='same', activation='relu'))
            if pooling:
                model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def lstm_model(units, input_shape=(1024,), num_classes=3862):
    """ Create a LSTM model.

    Parameters
    ----------
    units: int, list
        Number of units in convolution layer. If given an iterable,
        then will create different layers for each units.
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.

    Returns
    -------
    model: Model
        A LSTM model
    """

    model = Sequential()
    model.add(Reshape((1, input_shape[0]), input_shape=input_shape))
    if isinstance(units, int):
        model.add(LSTM(units))
    else:
        for unit in units:
            model.add(LSTM(unit))

    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def gru_model(units, input_shape=(1024,), num_classes=3862):
    """ Create a GRU model.

    Parameters
    ----------
    units: int, list
        Number of units in convolution layer. If given an iterable,
        then will create different layers for each units.
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.

    Returns
    -------
    model: Model
        A GRU model
    """

    model = Sequential()
    model.add(Reshape((1, input_shape[0]), input_shape=input_shape))
    if isinstance(units, int):
        model.add(GRU(units))
    else:
        for unit in units:
            model.add(GRU(unit))

    model.add(Dense(num_classes, activation='sigmoid'))
    return model


def create_model(units, choice='cnn', input_shape=(1024,), pooling=True):
    """ Create a model given the model choice. Pooling only effects
    CNN model.

    Parameters
    ----------
    units: int, list
        Number of units in each layer
    choice: str
        Available choices are cnn, lstm, gru and dnn.
    input_shape: tuple, list
        Input shape of the model
    pooling: bool
        Apply max pool after conv layer. Effects only cnn model

    Returns
    -------
    model: Model
        Tensorflow keras Model instance
    """

    if choice is 'cnn':
        model = cnn_model(units=units, input_shape=input_shape, pooling=pooling)
    elif choice is 'lstm':
        model = lstm_model(units=units, input_shape=input_shape)
    elif choice is 'gru':
        model = gru_model(units=units, input_shape=input_shape)
    else:  # DNN model
        model = dnn_model(units=units, input_shape=input_shape)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[top5_acc, top1_acc])
    return model


def fine_tune_model(train_records, val_records, sel, parser='example',
                    train=True, repeats=1000, cores=4, batch_size=32,
                    buffer_size=1, num_classes=3862, units=None, choice='cnn',
                    input_shape=(1024,), pooling=True, tensorboard=True):
    """

    Parameters
    ----------
    train_records: str
        A list of strings of TFRecord train file names
    val_records: str
        A list of strings of TFRecord validation file names
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
    units: int, list
        Number of units in each layer
    choice: str
        Available choices are cnn, lstm, gru and dnn.
    input_shape: tuple, list
        Input shape of the model
    pooling: bool
        Apply max pool after conv layer. Effects only cnn model
    tensorboard: bool
        Whether to output tensorboard logs. Default is true

    Returns
    -------
    model_history:
        A dict contains model history objects.

    """

    if choice not in ['cnn', 'lstm', 'dnn', 'gru']:
        raise ('Selected model is not supported yet. \
                Please select a valid model')

    # Grid Params
    _units = [128, 256, 512, 1024,  # 1 layer
              [128, 256], [256, 512], [512, 1024],  # 2 layers
              [128, 256, 512], [256, 512, 1024],  # 3 layers
              [128, 256, 512, 1024]]  # 4 layers

    layer_units = _units if units is None else units
    model_history = {}
    stop = EarlyStopping(patience=3)
    callbacks = [stop]

    for c, unit in enumerate(layer_units):
        train_data = get_data(records=train_records, sel=sel, parser=parser,
                              train=train, repeats=repeats, cores=cores,
                              batch_size=batch_size, buffer_size=buffer_size,
                              num_classes=num_classes)

        val_data = get_data(records=val_records, sel=sel, parser=parser,
                            train=train, repeats=repeats, cores=cores,
                            batch_size=batch_size, buffer_size=buffer_size,
                            num_classes=num_classes)

        model = create_model(units=unit, choice=choice,
                             input_shape=input_shape, pooling=pooling)
        if tensorboard:
            model_format = {
                'TYPE': choice.upper(),
                'LOOPS': c}
            MODEL_NAME = '{TYPE}_{LOOPS}'.format(**model_format)
            board = TensorBoard(log_dir='./logs/{}'.format(MODEL_NAME))
            callbacks.append(board)

        history = model.fit(x=train_data, steps_per_epoch=30, epochs=20,
                            validation_data=val_data, validation_steps=8,
                            verbose=0, callbacks=callbacks)

        model_history['model_' + str(c+1)] = history
        clear_session()
        callbacks.remove(board)

    return model_history

