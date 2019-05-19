import matplotlib.pyplot as plt
import pickle


def plot_loss(choice, sel, legends, loc='upper left'):
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

    train_top1_acc = {}
    train_top5_acc = {}
    val_top1_acc = {}
    val_top5_acc = {}

    for key, value in choice.items():
        if key.startswith(sel):
            if key.find('VAL') > 0:
                if key.find('TOP1') > 0:
                    val_top1_acc[key] = value
                else:
                    val_top5_acc[key] = value
            else:
                if key.find('TOP1') > 0:
                    train_top1_acc[key] = value
                else:
                    train_top5_acc[key] = value

    fig = plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(221)
    for key in val_top1_acc.keys():
        ax.plot(val_top1_acc[key])
    ax.set_title('Model Validation Hit@1 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(222)
    for key in train_top1_acc.keys():
        ax.plot(train_top1_acc[key])
    ax.set_title('Model Training Hit@1 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(223)
    for key in val_top5_acc.keys():
        ax.plot(val_top5_acc[key])
    ax.set_title('Model Validation Hit@5 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(224)
    for key in train_top5_acc.keys():
        ax.plot(train_top5_acc[key])
    ax.set_title('Model Training Hit@5 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    plt.tight_layout()
    plt.show()


def plot_params(choice, sel, legends, loc='upper left'):
    """ Plots parameter number with models together.

    Parameters
    ----------
    choice: dict
        A dict contains different models
    legends: list
        A list contains plot legends
    loc: str
        Legend position. Default is upper left.
    """

    plt.figure(figsize=(16, 8))
    values = [value for key, value in choice.items()]
    plt.bar(list(choice.keys()), values)
    plt.title('Number of Parameters in Models')
    plt.tick_params(axis='x', rotation=45)
    plt.ylabel('Count')
    plt.xlabel('Model Type')
    plt.legend(legends, loc=loc)
    plt.show()


def dump_results(models, params, fname='results.pickle'):
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

        accs[name + '_VAL_TOP1_ACC'] = model.history['val_top1_acc']
        accs[name + '_VAL_TOP5_ACC'] = model.history['val_top5_acc']
        accs[name + '_TOP1_ACC'] = model.history['top1_acc']
        accs[name + '_TOP5_ACC'] = model.history['top5_acc']

    dump = {'losses' : losses, 'accs': accs, 'params': params}
    with open(fname, 'wb') as f:
        pickle.dump(dump, f)