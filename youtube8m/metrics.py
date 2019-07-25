from sklearn.metrics import precision_recall_curve, label_ranking_average_precision_score
from sklearn.metrics.base import _average_binary_score
from youtube8m.metric_utils import AveragePrecisionCalculator as AP
from youtube8m.metric_utils import MeanAveragePrecisionCalculator as mAP
import numpy as np
import functools


class EvaluationMetrics(object):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Parameters
    ----------
    num_class: int
        A positive integer specifying the number of classes.
    top_k: int
        A positive integer specifying how many predictions are considered per video.
    top_n: int, None
        A positive Integer specifying the average precision at n, or None
        to use all provided data points.

    Raises
    ------
    ValueError:
        An error occurred when MeanAveragePrecisionCalculator cannot not be constructed.
    """

    def __init__(self, num_class, top_k, top_n=None):
        self.sum_hit_at_one = 0.0
        self.sum_perr = 0.0
        self.sum_precision = 0.0
        self.sum_lrap = 0.0
        self.mean_ap = mAP(num_class, top_n=top_n)
        self.global_ap = AP()
        self.top_k = top_k
        self.num_examples = 0

    @staticmethod
    def _flatten(l):
        """ Merges a list of lists into a single list."""
        return [item for sublist in l for item in sublist]

    @staticmethod
    def _top_k_by_class(predictions, labels, k=20):
        """Extracts the top k predictions for each video, sorted by class.

        Parameters
        -----------
        predictions: np.ndarray
            A numpy matrix containing the outputs of the model. Dimensions are 'batch' x 'num_classes'.
        labels: np.ndarray
            A numpy matrix containing the ground truths. Dimensions are 'batch' x 'num_classes'.
        k: int
            The top k non-zero entries to preserve in each prediction.

        Returns
        --------
        predictions,labels, true_positives: (list, list, list)
            A tuple 'predictions' and 'labels' are lists of lists of floats.
            'true_positives' is a list of scalars. The length of the lists are equal
            to the number of classes. The entries in the predictions variable are
            probability predictions, and the corresponding entries in the labels
            variable are the ground truth for those predictions. The entries in
            'true_positives' are the number of true positives for each class in the ground truth.

        Raises
        -------
        ValueError:
            An error occurred when the k is not a positive integer.
        """

        def top_k_triplets(y_pred, y_true, _k=20):
            """Get the top_k for a 1-d ndarrays."""
            _k = min(_k, len(y_pred))
            indices = np.argpartition(y_pred, -_k)[-_k:]
            return [(index, y_pred[index], y_true[index]) for index in indices]

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        num_classes = predictions.shape[1]
        k = min(k, num_classes)
        triplets = []
        for batch in range(predictions.shape[0]):
            triplets.extend(top_k_triplets(predictions[batch], labels[batch], k))

        _predictions = [[] for _ in range(num_classes)]
        _labels = [[] for _ in range(num_classes)]

        for triplet in triplets:
            _predictions[triplet[0]].append(triplet[1])
            _labels[triplet[0]].append(triplet[2])

        _true_positives = [np.sum(labels[:, i]) for i in range(num_classes)]

        return _predictions, _labels, _true_positives

    @staticmethod
    def hit_at_one(predictions, labels):
        """Performs a local (numpy) calculation of the hit at one.

        Parameters
        -----------
        predictions: np.ndarray
            Matrix containing the outputs of the model. Dimensions are 'batch' x 'num_classes'.
        labels: np.ndarray
            Matrix containing the ground truth labels. Dimensions are 'batch' x 'num_classes'.

        Returns
        --------
        avg_hits: float
            The average hit at one across the entire batch.
        """

        top_prediction = np.argmax(predictions, 1)
        hits = labels[np.arange(labels.shape[0]), top_prediction]
        return np.average(hits)

    @staticmethod
    def precision_at_equal_recall_rate(predictions, labels):
        """Performs a local calculation of the PERR.

        Parameters
        -----------
        predictions: np.ndarray
            Matrix containing the outputs of the model. Dimensions are 'batch' x 'num_classes'.
        labels: np.ndarray
            Matrix containing the ground truth labels. Dimensions are 'batch' x 'num_classes'.

        Returns
        -------
        agg_precision: float
            The average precision at equal recall rate across the entire batch.
        """

        aggregated_precision = 0.0
        batches = labels.shape[0]

        for i in range(batches):
            num_labels = int(np.sum(labels[i]))
            top_indices = np.argpartition(predictions[i], -num_labels)[-num_labels:]
            item_precision = 0.0
            for label_index in top_indices:
                if predictions[i][label_index] > 0:
                    item_precision += labels[i][label_index]

            item_precision /= top_indices.size
            aggregated_precision += item_precision

        aggregated_precision /= batches
        return aggregated_precision

    @staticmethod
    def average_precision_score(y_true, y_score, average="macro", pos_label=1, sample_weight=None):
        def _binary_uninterpolated_average_precision(y_true, y_score, pos_label=1, sample_weight=None):
            precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label, sample_weight)
            recall[np.isnan(recall)] = 0
            return -np.sum(np.diff(recall) * np.array(precision)[:-1])

        average_precision = functools.partial(_binary_uninterpolated_average_precision, pos_label=pos_label)

        return _average_binary_score(average_precision, y_true, y_score, average, sample_weight)

    def global_average_precision(self, predictions, labels, top_k=20):
        """Performs a local calculation of the global average precision.
        Only the top_k predictions are taken for each of the videos.

        Parameters
        ----------
        predictions: np.ndarray
            Matrix containing the outputs of the model. Dimensions are 'batch' x 'num_classes'.
        labels: np.ndarray
            Matrix containing the ground truth labels. Dimensions are 'batch' x 'num_classes'.
        top_k: int
            How many predictions to use per video.

        Returns
        -------
        gap: float
            The global average precision.
        """

        gap = AP()
        sparse_predictions, sparse_labels, num_positives = self._top_k_by_class(predictions, labels, top_k)
        gap.store(self._flatten(sparse_predictions), self._flatten(sparse_labels), sum(num_positives))
        return gap.peek_ap_at_n()

    def store(self, predictions, labels):
        """Accumulate the metrics calculated locally for this mini-batch.

        Parameters
        -----------
        predictions: np.ndarray
            A numpy matrix containing the outputs of the model. Dimensions are 'batch' x 'num_classes'.
        labels: np.ndarray
            A numpy matrix containing the ground truth labels. Dimensions are 'batch' x 'num_classes'.

        Returns
        --------
        metrics: tuple
            A tuple for mean hit@1 and mean PERR metrics for the mini-batch.

        Raises
        -------
        ValueError:
            An error occurred when the shape of predictions and labels does not match.
        """

        batch_size = labels.shape[0]
        mean_hit_at_one = self.hit_at_one(predictions, labels)
        mean_perr = self.precision_at_equal_recall_rate(predictions, labels)
        mean_lrap = label_ranking_average_precision_score(labels, predictions)
        mean_precision = self.average_precision_score(labels, predictions, average='weighted')

        # Take the top 20 predictions.
        sparse_predictions, sparse_labels, num_positives = self. _top_k_by_class(predictions, labels,
                                                                                 self.top_k)
        self.mean_ap.store(sparse_predictions, sparse_labels, num_positives)
        self.global_ap.store(self._flatten(sparse_predictions), self._flatten(sparse_labels), sum(num_positives))
        self.num_examples += batch_size
        self.sum_hit_at_one += mean_hit_at_one * batch_size
        self.sum_perr += mean_perr * batch_size
        self.sum_lrap += mean_lrap * batch_size
        self.sum_precision += mean_precision * batch_size

        return mean_hit_at_one, mean_perr, mean_precision

    def eval(self):
        """Calculate the evaluation metrics for the whole epoch.

        Raises
        ------
        ValueError:
            If no examples were accumulated.

        Returns
        --------
        metrics: dict
            A dictionary storing the evaluation metrics for the epoch. The dictionary
            has the fields: avg_hit_at_one, avg_perr, aps (default nan), gap, lrap and precision.
        """

        if self.num_examples <= 0:
            raise ValueError("total_sample must be positive.")

        avg_hit_at_one = self.sum_hit_at_one / self.num_examples
        avg_perr = self.sum_perr / self.num_examples
        avg_lrap = self.sum_lrap / self.num_examples
        avg_precision = self.sum_precision / self.num_examples

        aps = self.mean_ap.peek_map_at_n()
        gap = self.global_ap.peek_ap_at_n()

        metrics = {
            "avg_hit_at_one": avg_hit_at_one,
            "avg_perr": avg_perr,
            "aps": aps,
            "gap": gap,
            "lrap": avg_lrap,
            "precision": avg_precision,
        }

        return metrics

    def clear(self):
        """Clear the evaluation metrics and reset the EvaluationMetrics object."""
        self.sum_hit_at_one = 0.0
        self.sum_perr = 0.0
        self.mean_ap.clear()
        self.global_ap.clear()
        self.num_examples = 0
