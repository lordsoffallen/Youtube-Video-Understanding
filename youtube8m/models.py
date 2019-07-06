from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.backend import sum
from tensorflow.python.keras.layers import Dense, Flatten, LSTM, GRU, Activation, Dropout
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, CuDNNLSTM, CuDNNGRU
from tensorflow.python.keras.layers import BatchNormalization, Add, Reshape, Multiply, Lambda
from tensorflow.python.keras.regularizers import l2


class BaseModel(object):
    def create(self):
        raise NotImplementedError()


class MLPModel(BaseModel):
    """Create a Multi Layer Perceptron model. Use create() to build the model

    Parameters
    ----------
    units: int, list
        Number of units in dense layer. If given an iterable,
        then will create different layers for each units.
    include_top: bool
        If true, adds the classification layer using number of classes
    last_activation: str
        Only used when include_top is true. Defines last layer activation
    batch_normalization: bool
        Whether to add batch normalization after dense layers. Ignored when
        units parameter is int
    kernel_regularizer:
        Kernel weights regularizer
    dropout: bool
        Whether to add dropout after dense layers. If true, it adds dropout
        with a 0.25 rate. Ignored when units parameter is int.
    summary: bool
        Whether to print the summary of model
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.
    """

    def __init__(self, units,
                 include_top=True,
                 last_activation='sigmoid',
                 batch_normalization=False,
                 kernel_regularizer=l2(1e-8),
                 dropout=False,
                 summary=False,
                 input_shape=(1024,),
                 num_classes=3862):

        self.units = units
        self.include_top = include_top
        self.last_activation = last_activation
        self.batch_normalization = batch_normalization
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        self.dropout_rate = 0.25
        self.summary = summary
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create(self):
        """ Creates MLP model.

        Returns
        -------
        model: Model
            A Dense Neural Network model
        """

        model = Sequential()
        if isinstance(self.units, int):
            model.add(Dense(self.units, activation='relu',
                            input_shape=self.input_shape,
                            kernel_regularizer=self.kernel_regularizer))
        else:  # Retrieve first unit and add input shape
            model.add(Dense(self.units.pop(0),
                            input_shape=self.input_shape,
                            kernel_regularizer=self.kernel_regularizer))
            if self.batch_normalization:
                model.add(BatchNormalization())
                model.add(Activation('relu'))
            elif self.dropout:
                model.add(Activation('relu'))
                model.add(Dropout(self.dropout_rate))
            else:  # No norm or dropout
                model.add(Activation('relu'))
                for unit in self.units:
                    model.add(Dense(unit, kernel_regularizer=self.kernel_regularizer))
                    if self.batch_normalization:
                        model.add(BatchNormalization())
                        model.add(Activation('relu'))
                    elif self.dropout:
                        model.add(Activation('relu'))
                        model.add(Dropout(self.dropout_rate))
                    else:
                        model.add(Activation('relu'))
        if self.include_top:
            model.add(Dense(self.num_classes,
                            activation=self.last_activation,
                            kernel_regularizer=self.kernel_regularizer))
        if self.summary:
            model.summary()
        return model


class LogisticModel(BaseModel):
    """Init Logistic regression model. Use create() to build the model

    Parameters
    ----------
    last_activation: str
        Only used when include_top is true. Defines last layer activation
    kernel_regularizer:
        Kernel weights regularizer
    summary: bool
        Whether to print the summary of model
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.
    """

    def __init__(self, last_activation='sigmoid',
                 kernel_regularizer=l2(1e-8),
                 summary=False,
                 input_shape=(1024,),
                 num_classes=3862):

        self.last_activation = last_activation
        self.kernel_regularizer = kernel_regularizer
        self.summary = summary
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create(self):
        """ Creates the logistic model.

        Returns
        -------
        model: Model
            A Logistic regression model
        """

        model = Sequential()
        model.add(Dense(self.num_classes, kernel_regularizer=self.kernel_regularizer,
                        activation=self.last_activation, input_shape=self.input_shape))
        if self.summary:
            model.summary()
        return model


class ResNetModel(BaseModel):
    """Init Residual Network model using Dense layers. Use create() to build the model

    Parameters
    ----------
    units: list
        Number of units in each block layers.
    include_top: bool
        If true, adds the classification layer using number of classes
    last_activation: str
        Only used when include_top is true. Defines last layer activation
    kernel_regularizer:
        Kernel weights regularizer
    summary: bool
        Whether to print the summary of model
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.
    """

    def __init__(self, units=None,
                 include_top=True,
                 last_activation='softmax',
                 kernel_regularizer=l2(1e-8),
                 summary=False,
                 input_shape=(1024,),
                 num_classes=3862):
        if isinstance(units, int):
            raise TypeError('ResNet Model takes units a list argument. Be sure'
                            'to pass multiple units as this model specifically '
                            'requires 3 block therefore 3 int unit arguments')

        self.units = [512, 256, 512] if units is None else units
        self.include_top = include_top
        self.last_activation = last_activation
        self.kernel_regularizer = kernel_regularizer
        self.summary = summary
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _residual_block(self, input_tensor, u1, u2, u3):
        # TODO Investigate the order of relu and batchnorm
        i = Dense(u1, kernel_regularizer=self.kernel_regularizer)(input_tensor)
        i = Activation('relu')(i)
        i = BatchNormalization()(i)

        i = Dense(u2, kernel_regularizer=self.kernel_regularizer)(i)
        i = Activation('relu')(i)
        i = BatchNormalization()(i)

        i = Dense(u3, kernel_regularizer=self.kernel_regularizer)(i)
        o = Add()([i, input_tensor])
        return o

    def create(self):
        """ Create a ResNet model

        Returns
        -------
        model: Model
            A ResNet model instance
        """
        inputs = Input(shape=self.input_shape)
        last_dim = self.input_shape[0]
        u1 = self.units[0]
        u2 = self.units[1]
        u3 = self.units[2]

        x = self._residual_block(inputs, u1, u1, last_dim)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = self._residual_block(x, u2, u2, last_dim)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = self._residual_block(x, u3, u3, last_dim)
        x = Activation('relu')(x)

        if self.include_top:    # TODO check sigmoid activation
            x = Dense(self.num_classes,
                      activation=self.last_activation,
                      kernel_regularizer=self.kernel_regularizer)(x)
        model = Model(inputs, x)
        if self.summary:
            model.summary()
        return model


class MixtureOfExpertsModel(BaseModel):
    """Create a Mixture of Experts model. Model consists of a per-class softmax
    distribution over a configurable number of logistic classifiers. One of the
    classifiers in the mixture is not trained, and always predicts 0. If unit is given
    then it becomes MoE with 2 layer.

    Use create() to build the model

    Parameters
    ----------
    units: int
        If given, this adds another layer to process input data.
    num_experts: int
        Number of experts excluding a dummy 'expert' that always
        predicts the non-existence of an entity
    kernel_regularizer:
        Kernel weights regularizer
    summary: bool
        Whether to print the summary of model
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.
    """

    def __init__(self, units=None,
                 num_experts=2,
                 kernel_regularizer=l2(1e-8),
                 summary=False,
                 input_shape=(1024,),
                 num_classes=3862):

        self.units = units
        self.num_experts = num_experts
        self.kernel_regularizer = kernel_regularizer
        self.summary = summary
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create(self):
        """ Creates MoE model.

        Returns
        -------
        model: Model
            A Mixture of Experts model
        """

        inputs = Input(shape=self.input_shape)
        if self.units is not None:
            gate_activations = Dense(self.units, kernel_regularizer=self.kernel_regularizer)(inputs)
            gate_activations = Dense(self.num_classes * (self.num_experts + 1),
                                     kernel_regularizer=self.kernel_regularizer)(gate_activations)
        else:
            gate_activations = Dense(self.num_classes * (self.num_experts + 1),
                                     kernel_regularizer=self.kernel_regularizer)(inputs)

        expert_activations = Dense(self.num_classes * self.num_experts,
                                   kernel_regularizer=self.kernel_regularizer)(inputs)

        # (Batch * #Labels) x (num_experts + 1)
        gate_reshaped = Reshape((self.num_classes, self.num_experts + 1))(gate_activations)
        gating_distribution = Activation('softmax')(gate_reshaped)

        # (Batch * #Labels) x num_experts
        expert_reshaped = Reshape((self.num_classes, self.num_experts))(expert_activations)
        expert_distribution = Activation('sigmoid')(expert_reshaped)

        slice_gating = Lambda(lambda x: x[:, :, :self.num_experts])(gating_distribution)
        probs = Multiply()([slice_gating, expert_distribution])

        outputs = Lambda(lambda x: sum(x, axis=2))(probs)
        model = Model(inputs, outputs)

        if self.summary:
            model.summary()

        return model


class CNNModel(BaseModel):
    """Create a Convolutional Neural Network model. Stackes Conv1D with
    pool layers (if given) together. Use create() to build the model

    Parameters
    ----------
    filters: int, list
        Number of filters in conv1d layer. If given an iterable,
        then will create different layers for each units.
    kernel_size: int
        Kernel size of the conv. Default is 3
    strides: int
        Strides size of conv. Default is 1
    pool: int
        Whether to add pooling after conv layer or not. If 0 no pooling will be added.
    include_top: bool
        If true, adds the classification layer using number of classes
    last_activation: str
        Only used when include_top is true. Defines last layer activation
    batch_normalization: bool
        Whether to add batch normalization after conv1d layers. Ignored when
        units parameter is int
    kernel_regularizer:
        Kernel weights regularizer
    dropout: bool
        Whether to add dropout after conv1d layers. If true, it adds dropout
        with a 0.25 rate. Ignored when units parameter is int.
    summary: bool
        Whether to print the summary of model
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.
    """

    def __init__(self, filters,
                 kernel_size=3,
                 strides=1,
                 pool=2,
                 include_top=True,
                 last_activation='sigmoid',
                 batch_normalization=False,
                 kernel_regularizer=l2(1e-8),
                 dropout=False,
                 summary=False,
                 input_shape=(1024,),
                 num_classes=3862):

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool = pool
        self.include_top = include_top
        self.last_activation = last_activation
        self.batch_normalization = batch_normalization
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        self.dropout_rate = 0.25
        self.summary = summary
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create(self):
        """ Creates CNN model.

        Returns
        -------
        model: Model
            A Convolutinal Neural Network model
        """

        model = Sequential()
        model.add(Reshape((self.input_shape[0], 1), input_shape=self.input_shape))

        if isinstance(self.filters, int):
            model.add(Conv1D(self.filters, self.kernel_size,
                             strides=self.strides,
                             padding='valid',
                             activation='relu',
                             input_shape=self.input_shape,
                             kernel_regularizer=self.kernel_regularizer))
            if self.pool > 0:
                model.add(MaxPooling1D(self.pool))
        else:
            for filter in self.filters:
                model.add(Conv1D(filter, self.kernel_size,
                                 strides=self.strides,
                                 padding='valid',
                                 input_shape=self.input_shape,
                                 kernel_regularizer=self.kernel_regularizer))
                if self.batch_normalization:
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                elif self.dropout:
                    model.add(Activation('relu'))
                    model.add(Dropout(self.dropout_rate))
                else:
                    model.add(Activation('relu'))

                if self.pool > 0:
                    model.add(MaxPooling1D(self.pool))

        model.add(Flatten())
        if self.include_top:
            model.add(Dense(self.num_classes,
                            activation=self.last_activation,
                            kernel_regularizer=self.kernel_regularizer))
        if self.summary:
            model.summary()
        return model


class RNNModel(BaseModel):
    """Create a Recurrent Neural Network model. Stackes either LSTM or
     GRU cells together. Use create() to build the model

    Parameters
    ----------
    units: int, list
        Number of units in lstm or gru layer. If given an iterable,
        then will create different layers for each units.
    gpu: bool
        If GPU available then use GPU implementation of layers.
    include_top: bool
        If true, adds the classification layer using number of classes
    last_activation: str
        Only used when include_top is true. Defines last layer activation
    batch_normalization: bool
        Whether to add batch normalization after lstm-gru layers. Ignored when
        units parameter is int
    kernel_regularizer:
        Kernel weights regularizer
    dropout: bool
        Whether to add dropout after lstm-gru layers. If true, it adds dropout
        with a 0.25 rate. Ignored when units parameter is int.
    summary: bool
        Whether to print the summary of model
    input_shape: tuple, list
        Input shape of the model
    num_classes: int
        Number of output classes.
    """

    def __init__(self, units,
                 cell_type='lstm',
                 include_top=True,
                 last_activation='sigmoid',
                 batch_normalization=False,
                 kernel_regularizer=l2(1e-8),
                 dropout=False,
                 gpu=False,
                 summary=False,
                 input_shape=(1024,),
                 num_classes=3862):

        self.units = units
        self.cell_type = cell_type
        self.gpu = gpu
        self.include_top = include_top
        self.last_activation = last_activation
        self.batch_normalization = batch_normalization
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        self.dropout_rate = 0.25
        self.summary = summary
        self.input_shape = input_shape
        self.num_classes = num_classes

        if self.gpu:
            if self.cell_type is 'lstm':
                self.LAYER_TYPE = CuDNNLSTM
            else:
                self.LAYER_TYPE = CuDNNGRU
        else:
            if self.cell_type is 'lstm':
                self.LAYER_TYPE = LSTM
            else:
                self.LAYER_TYPE = GRU

    def create(self):
        """ Creates RNN model.

        Returns
        -------
        model: Model
            A Recurrent Neural Network model
        """

        model = Sequential()
        model.add(Reshape((self.input_shape[0], 1), input_shape=self.input_shape))

        if isinstance(self.units, int):
            model.add(self.LAYER_TYPE(self.units, kernel_regularizer=self.kernel_regularizer))
        else:
            for c, unit in enumerate(self.units, start=1):
                if c == len(self.units):  # Last element. Return sequence is false
                    model.add(self.LAYER_TYPE(unit, activation=None,
                                              kernel_regularizer=self.kernel_regularizer))
                else:
                    model.add(self.LAYER_TYPE(unit, activation=None,
                                              kernel_regularizer=self.kernel_regularizer,
                                              return_sequences=True))

                if self.batch_normalization:
                    model.add(BatchNormalization())
                    model.add(Activation('tanh'))
                elif self.dropout:
                    model.add(Activation('tanh'))
                    model.add(Dropout(self.dropout_rate))
                else:
                    model.add(Activation('tanh'))

        if self.include_top:
            model.add(Dense(self.num_classes,
                            activation=self.last_activation,
                            kernel_regularizer=self.kernel_regularizer))
        if self.summary:
            model.summary()
        return model

