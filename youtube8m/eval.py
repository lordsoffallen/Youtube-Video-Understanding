from tensorflow.python.keras.backend import get_session
from tensorflow.python.data import TFRecordDataset
from tensorflow.python.framework.errors import OutOfRangeError
from tensorflow.python.keras import Model
from youtube8m.metrics import EvaluationMetrics
import pickle
import warnings

warnings.filterwarnings('ignore')   # Ignore sklearn true_divide warning


def eval_model(model, val_dataset, top_k=20, num_classes=3862):
    """Evaluate model performance on validation dataset. This function
    calculates the hit@1, PERR, AP, LRAP, GAP and mAP values.

    Parameters
    ----------
    model: Model
        Keras model instance to measure performance
    val_dataset: TFRecordDataset
        A dataset contains record files.
    top_k: int
        A positive integer specifying how many predictions are considered per video.
    num_classes: int
        A positive integer specifying the number of classes.
    """

    # Create metric class
    metrics = EvaluationMetrics(num_classes, top_k=top_k)

    # Get session
    sess = get_session()

    # Create an iterator
    ds_iter = val_dataset.make_one_shot_iterator()
    batched_data = ds_iter.get_next()

    print('Calculating metrics(hit@1 and PERR) per mini batch...')
    while True:
        try:
            X, y = sess.run(batched_data)
            preds = model.predict(X)
            hit1, perr, precision = metrics.store(preds, y)
            print('Hit@1: {:.3f} PERR: {:.3f} AP: {:.3f}'.format(hit1, perr, precision), end='\r')

        except OutOfRangeError:
            break

    print('Calculating global metrics.......')
    gmetrics = metrics.eval()

    print('Hit@1: {:.3f} \n'
          'PERR(Precision at Equal Recall Rate): {:.3f} \n'
          'GAP(Global Average Precision): {:.3f} \n'
          'LRAP(Label Ranking Average Precision): {:.3f} \n'
          'AP(Average Precision): {:.3f} \n'
          .format(gmetrics['avg_hit_at_one'], gmetrics['avg_perr'], gmetrics['gap'],
                  gmetrics['lrap'], gmetrics['precision']))

    print('Saving mean average precision values based on top_k or per class...')
    mAp = gmetrics['aps']
    with open('pickles/mAP_values.pkl', 'wb') as f:
        pickle.dump(mAp, f)
    print('Mean Average Precisions are saved to mAP_values.pkl.')
    metrics.clear()

