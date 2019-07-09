import matplotlib.pyplot as plt
import pickle


def plot_loss(choice, sel, legends, loc='upper right'):
    """ Plots validation loss and training loss together.

    Parameters
    ----------
    choice: dict
        A dict contains different models
    sel: str
        Defines which model to plot. Options are DNN, CNN, LSTM, GRU
    legends: list
        A list contains plot legends
    loc: str
        Legend position. Default is upper left.
    """

    train_loss = {}
    val_loss = {}
    for key, value in choice.items():
        if key.startswith(sel):
            if key.find('VAL') > 0:
                val_loss[key] = value
            else: # Train loss
                train_loss[key] = value

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    for key in val_loss.keys():
        ax.plot(val_loss[key])
    ax.set_title('Model Validation Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(122)
    for key in train_loss.keys():
        ax.plot(train_loss[key])
    ax.set_title('Model Training Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    plt.tight_layout()
    plt.show()


def plot_acc(choice, sel, legends, loc='upper left'):
    """ Plots validation and training's Top 1 and Top 5 accuracy together.

    Parameters
    ----------
    choice: dict
        A dict contains different models
    sel: str
        Defines which model to plot. Options are DNN, CNN, LSTM, GRU
    legends: list
        A list contains plot legends
    loc: str
        Legend position. Default is upper left.
    """

    train_top5_acc = {}
    val_top5_acc = {}

    for key, value in choice.items():
        if key.startswith(sel):
            if key.find('VAL') > 0:
                val_top5_acc[key] = value
            else:
                train_top5_acc[key] = value

    fig = plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(121)
    for key in val_top5_acc.keys():
        ax.plot(val_top5_acc[key])
    ax.set_title('Model Validation Hit@5 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(122)
    for key in train_top5_acc.keys():
        ax.plot(train_top5_acc[key])
    ax.set_title('Model Training Hit@5 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    plt.tight_layout()
    plt.show()


def plot_params(choice):
    """ Plots parameter number with models together.

    Parameters
    ----------
    choice: dict
        A dict contains different models
    """

    plt.figure(figsize=(16, 8))
    values = [value for key, value in choice.items()]
    plt.bar(list(choice.keys()), values)
    plt.title('Number of Parameters in Models')
    plt.tick_params(axis='x', rotation=45)
    plt.ylabel('Count')
    plt.xlabel('Model Type')
    plt.show()


def dump_results(models, params, fname='results.pkl'):
    """ Dumps the results into a pickle object.

    Parameters
    ----------
    models: dict
        A dict contains model history object
    params: dict
        A dict contains model parameters
    fname: str
        File name to write results.
    """

    losses = {}
    accs = {}

    # Collect all losses and accs from models
    for name, model in models.items():
        losses[name + '_VAL_LOSS'] = model.history['val_loss']
        losses[name + '_LOSS'] = model.history['loss']

        accs[name + '_VAL_HIT1'] = model.history['val_hit1']
        accs[name + '_HIT1'] = model.history['hit1']

        accs[name + '_VAL_HIT3'] = model.history['val_hit3']
        accs[name + '_HIT3'] = model.history['hit3']

    dump = {'losses': losses, 'accs': accs, 'params': params}
    with open(fname, 'wb') as f:
        pickle.dump(dump, f)
