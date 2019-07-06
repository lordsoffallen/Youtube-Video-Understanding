from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.losses import huber_loss
from tensorflow.python.data import TFRecordDataset
from .models import MLPModel, CNNModel, RNNModel, MixtureOfExpertsModel, LogisticModel, ResNetModel, Model
from .metrics import top5_acc
from .train_utils import HammingLoss


def create_model(units=None, choice='mlp', loss_fn='huber', input_shape=(1024,), **kwargs):
    """Create a model given the model choice. Keywords args contains model specific
    arguments. Look at the model implementation in models.py for more details
    Model will be compiled using binary crossentropy while using top5
    as metric.

    Parameters
    ----------
    units: int, list
        Number of units in each layer. Ignored when choice is logistic.
    choice: str
        Available choices are cnn, lstm, gru, dnn and logistic.(Case sensitive)
    loss_fn: str
        Loss function. Options are huber, binary (binary crossentropy),
        cat (categorical crossentropy) and hamming
    input_shape: tuple, list
        Input shape of the model
    kwargs:
        Model Specific arguments to pass related models when creating

    Returns
    -------
    model: Model
        Tensorflow keras Model instance
    """

    if choice not in ['cnn', 'lstm', 'mlp', 'gru', 'logistic', 'resnet', 'moe']:
        raise ValueError('Selected model is not supported yet. Please select a valid model')

    if choice is 'cnn':
        model = CNNModel(units, input_shape=input_shape, **kwargs).create()
    elif choice is 'mlp':
        model = MLPModel(units, input_shape=input_shape, **kwargs).create()
    elif choice is 'resnet':
        model = ResNetModel(units, input_shape=input_shape, **kwargs).create()
    elif choice is 'logistic':
        model = LogisticModel(input_shape=input_shape, **kwargs).create()
    elif choice is 'moe':
        model = MixtureOfExpertsModel(units, input_shape=input_shape, **kwargs).create()
    else:     # LSTM and GRU
        model = RNNModel(units, cell_type=choice, input_shape=input_shape, **kwargs).create()

    # Parse loss_fn
    if loss_fn is 'binary':
        loss = 'binary_crossentropy'
    elif loss_fn is 'huber':
        loss = huber_loss
    elif loss_fn is 'hamming':
        loss = HammingLoss()
    else:
        loss = 'categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=[top5_acc])
    return model


def train_model(model, train_data, val_data, steps_per_epoch=None, validation_steps=None,
                tensorboard=True, checkpoint=True, model_name='resnet'):
    """ Trains a keras model given a train and validation datasets. It will checkpoint the best
    model at each epoch.

    Parameters
    ----------
    model: Model
        A keras model instance to train data on
    train_data: TFRecordDataset
        A dataset contains training data
    val_data: TFRecordDataset
        A dataset contains validation data
    steps_per_epoch: int
        Number of steps required to complete one training part.
    validation_steps: int
        Number of steps required to complete validation part.
    tensorboard: bool
        Whether to output tensorboard logs. Default is true
    checkpoint: bool
        If true use checkpoint callback to store model
    model_name: str
        Model name in str format. Used when saving the model

    Returns
    -------
    history:
        Keras history object contains training history data
    """

    # stop = LossChecker(patience=500)
    stop = EarlyStopping(patience=3)
    callbacks = [stop]
    if checkpoint:
        path = "model-{epoch:02d}-{val_loss:.2f}.h5"
        checkpoint = ModelCheckpoint(path, verbose=1)
        callbacks.append(checkpoint)

    if tensorboard:
        board = TensorBoard(log_dir='./logs/')
        callbacks.append(board)

    history = model.fit(x=train_data, steps_per_epoch=steps_per_epoch, epochs=10,
                        validation_data=val_data, validation_steps=validation_steps,
                        verbose=1, callbacks=callbacks)

    if model_name is 'moe':
        model.save_weights(model_name+'_model_weights.h5')
    else:
        model.save(model_name+'_model.h5', include_optimizer=False)
    return history
