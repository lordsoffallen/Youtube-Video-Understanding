from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.losses import huber_loss
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.data import TFRecordDataset
from .models import MLPModel, CNNModel, RNNModel, MixtureOfExpertsModel
from .models import LogisticModel, ResNetModel, Model
from .train_utils import HammingLoss, hit_at_one, hit_at_n


def create_model(units=None, choice='mlp', loss_fn='huber', optimizer='adam', batch_size=32, **kwargs):
    """Create a model given the model choice. Keywords args contains model specific
    arguments. Look at the model implementation in models.py for more details
    Model will be compiled using binary crossentropy while using hit1 and hit 3
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
    optimizer: str, function
        Optimizer of the model.
    batch_size: int
        Batch size parameter to pass on to model metric
    kwargs:
        Model Specific arguments to pass related models when creating

    Returns
    -------
    model: Model
        Tensorflow keras Model instance
    """

    def hit1(y_true, y_pred, batch=batch_size):
        return hit_at_one(y_true, y_pred, batch)

    def hit3(y_true, y_pred, batch=batch_size, n=3):
        return hit_at_n(y_true, y_pred, batch, n)

    if choice not in ['cnn', 'lstm', 'mlp', 'gru', 'logistic', 'resnet', 'moe']:
        raise ValueError('Selected model is not supported yet. Please select a valid model')

    if choice == 'cnn':
        model = CNNModel(units, **kwargs).create()
    elif choice == 'mlp':
        model = MLPModel(units, **kwargs).create()
    elif choice == 'resnet':
        model = ResNetModel(units, **kwargs).create()
    elif choice == 'logistic':
        model = LogisticModel(**kwargs).create()
    elif choice == 'moe':
        model = MixtureOfExpertsModel(units, **kwargs).create()
    else:     # LSTM and GRU
        model = RNNModel(units, cell_type=choice, **kwargs).create()

    # Parse loss_fn
    if loss_fn == 'binary':
        loss = 'binary_crossentropy'
    elif loss_fn == 'huber':
        loss = huber_loss
    elif loss_fn == 'hamming':
        loss = HammingLoss()
    else:
        loss = 'categorical_crossentropy'

    if optimizer == 'sgd':
        opt = SGD(lr=0.5, momentum=0.9, decay=0.0, nesterov=True)
        model.compile(optimizer=opt, loss=loss, metrics=[hit1, hit3])
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=[hit1, hit3])
    return model


def train_model(model, train_data, val_data, steps_per_epoch=None, validation_steps=None,
                tensorboard=True, checkpoint=True, model_name='resnet'):
    """ Trains a keras model given a train and validation datasets. It will checkpoint the best
    model at each epoch. When model contains Lambda layers, checkpoint should be false.

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

    stop = EarlyStopping(patience=2)
    callbacks = [stop]
    if checkpoint:
        path = "model-{epoch:02d}-{val_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(path, verbose=1)
        callbacks.append(checkpoint)

    if tensorboard:
        board = TensorBoard(log_dir='./logs/')
        callbacks.append(board)

    history = model.fit(x=train_data, steps_per_epoch=steps_per_epoch, epochs=10,
                        validation_data=val_data, validation_steps=validation_steps,
                        verbose=1, callbacks=callbacks)

    if model_name.startswith('moe'):
        model.save_weights(model_name+'_weights.h5')
    else:
        model.save(model_name+'.h5', include_optimizer=False)
    return history
