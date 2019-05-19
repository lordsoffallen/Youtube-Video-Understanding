import matplotlib.pyplot as plt
import json


def plot_loss(choice, legends, loc='upper left'):
    """ Plots validation loss and training loss together.

    Parameters
    ----------
    choice: dict
        A dict contains different models
    legends: list
        A list contains plot legends
    loc: str
        Legend position. Default is upper left.
    """

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    for _, model in choice.items():
        ax.plot(model.history['val_loss'])
    ax.set_title('Model Validation Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(122)
    for _, model in choice.items():
        ax.plot(model.history['loss'])
    ax.set_title('Model Training Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    plt.tight_layout()
    plt.show()


def plot_acc(choice, legends, loc='upper left'):
    """ Plots validation and training's Top 1 and Top 5 accuracy together.

    Parameters
    ----------
    choice: dict
        A dict contains different models
    legends: list
        A list contains plot legends
    loc: str
        Legend position. Default is upper left.
    """

    fig = plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(221)
    for _, model in choice.items():
        ax.plot(model.history['val_top1_acc'])
    ax.set_title('Model Validation Hit@1 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(222)
    for _, model in choice.items():
        ax.plot(model.history['top1_acc'])
    ax.set_title('Model Training Hit@1 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(223)
    for _, model in choice.items():
        ax.plot(model.history['val_top5_acc'])
    ax.set_title('Model Validation Hit@5 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    ax = fig.add_subplot(224)
    for _, model in choice.items():
        ax.plot(model.history['top5_acc'])
    ax.set_title('Model Training Hit@5 Acc')
    ax.set_ylabel('Acc')
    ax.set_xlabel('Epoch')
    ax.legend(legends, loc=loc)

    plt.tight_layout()
    plt.show()


def dump_results(models, params, fname='results.json'):
    """ Dumps the results into a json object.

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
        losses[name + 'VAL_LOSS'] = model.history['val_loss']
        losses[name + 'LOSS'] = model.history['loss']

        accs[name + 'VAL_TOP1_ACC'] = model.history['val_top1_acc']
        accs[name + 'VAL_TOP5_ACC'] = model.history['val_top5_acc']
        accs[name + 'TOP1_ACC'] = model.history['top1_ccc']
        accs[name + 'TOP5_ACC'] = model.history['top5_acc']

    dump = {'losses' : losses, 'accs': accs, 'params': params}
    with open(fname, 'w') as f:
        json.dump(dump, f)