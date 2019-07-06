import heapq
import random
import numpy as np


class AveragePrecisionCalculator(object):
    """Calculate or keep track of the interpolated average precision.
    It provides an interface for calculating interpolated average precision for an
    entire list or the top-n ranked items.

    Example usages:
    1) Use it as a static function call to directly calculate average precision for
    a short ranked list in the memory.

    ```
    import random

    p = np.array([random.random() for _ in range(10)])
    a = np.array([random.choice([0, 1]) for _ in range(10)])

    ap = AveragePrecisionCalculator.ap(p, a)
    ```

    2) Use it as an object for long ranked list that cannot be stored in memory or
    the case where partial predictions can be observed at a time (Tensorflow
    predictions). In this case, we first call the function accumulate many times
    to process parts of the ranked list. After processing all the parts, we call
    peek_interpolated_ap_at_n.
    ```
    p1 = np.array([random.random() for _ in range(5)])
    a1 = np.array([random.choice([0, 1]) for _ in range(5)])
    p2 = np.array([random.random() for _ in range(5)])
    a2 = np.array([random.choice([0, 1]) for _ in range(5)])

    # interpolated average precision at 10 using 1000 break points
    calculator = AveragePrecisionCalculator(10)
    calculator.store(p1, a1)
    calculator.store(p2, a2)
    ap3 = calculator.peek_ap_at_n()
    ```
    """

    def __init__(self, top_n=None):
        """Construct an AveragePrecisionCalculator to calculate average precision.
        This class is used to calculate the average precision for a single label.

        Parameters
        -----------
        top_n: int, None
            A positive Integer specifying the average precision at n, or None to use all provided data points.

        Raises
        -------
        ValueError:
            An error occurred when the top_n is not a positive integer.
        """

        if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
            raise ValueError("top_n must be a positive integer or None.")

        self._top_n = top_n  # average precision at n
        self._total_positives = 0  # total number of positives have seen
        self._heap = []  # max heap of (prediction, label)

    @property
    def heap_size(self):
        """Gets the heap size maintained in the class."""
        return len(self._heap)

    @property
    def num_accumulated_positives(self):
        """Gets the number of positive samples that have been accumulated."""
        return self._total_positives

    def store(self, predictions, labels, num_positives=None):
        """Accumulate the predictions and their ground truth labels.
        After the function call, we may call peek_ap_at_n to actually calculate
        the average precision.

        Parameters
        -----------
        predictions: list, np.ndarray
            A list or np array storing the prediction scores.
        labels: list, np.ndarray
            A list or np array storing the ground truth labels. Any value larger than 0
            will be treated as positives, otherwise as negatives.
        num_positives: int, None
            If the 'predictions' and 'labels' inputs aren't complete, then it's
            possible some true positives were missed in them. In that case, you can
            provide 'num_positives' in order to accurately track recall.

        Raises
        -------
        ValueError:
            An error occurred when the format of the input is not the
            numpy 1-D array or the shape of predictions and labels does not match.
        """

        if len(predictions) != len(labels):
            raise ValueError("The shape of predictions and labels does not match.")

        if num_positives is not None:
            if num_positives < 0:
                raise ValueError("'num_positives' should be a positive integer.")

            self._total_positives += num_positives
        else:
            if isinstance(labels, np.ndarray):
                self._total_positives += np.size(np.where(labels > 1e-5))
            else:
                self._total_positives += np.size(np.where(np.array(labels) > 1e-5))

        topk = self._top_n
        heap = self._heap

        for i in range(np.size(predictions)):
            if topk is None or len(heap) < topk:
                heapq.heappush(heap, (predictions[i], labels[i]))
            else:
                if predictions[i] > heap[0][0]:  # heap[0] is the smallest
                    heapq.heappop(heap)
                    heapq.heappush(heap, (predictions[i], labels[i]))

    def clear(self):
        """Clear the accumulated predictions."""
        self._heap = []
        self._total_positives = 0

    def peek_ap_at_n(self):
        """Peek the non-interpolated average precision at n.

        Returns
        -------
        ap: float
            The non-interpolated average precision at n (default 0).
            If n is larger than the length of the ranked list, the average precision will be returned.
        """

        if self.heap_size <= 0:
            return 0

        predlists = np.array(list(zip(*self._heap)))

        ap = self.ap_at_n(predlists[0], predlists[1], n=self._top_n,
                          total_num_positives=self._total_positives)
        return ap

    def ap(self, predictions, labels):
        """Calculate the non-interpolated average precision.

        Parameters
        -----------
        predictions: np.ndarray
            A numpy 1-D array storing the sparse prediction scores.
        labels: np.ndarray
            A numpy 1-D array storing the ground truth labels. Any value
            larger than 0 will be treated as positives, otherwise as negatives.

        Returns
        -------
        ap: float
            The non-interpolated average precision at n. If n is larger than the
            length of the ranked list, the average precision will be returned.

        Raises
        -------
        ValueError:
            An error occurred when the format of the input is not the
            numpy 1-D array or the shape of predictions and labels does not match.
        """

        return self.ap_at_n(predictions, labels, n=None)

    @staticmethod
    def ap_at_n(predictions, labels, n=20, total_num_positives=None):
        """Calculate the non-interpolated average precision.

        Parameters
        -----------
        predictions: np.ndarray
            A numpy 1-D array storing the sparse prediction scores.
        labels: np.ndarray
            A numpy 1-D array storing the ground truth labels.
            Any value larger than 0 will be treated as positives, otherwise as negatives.
        n: int, None
            The top n items to be considered in ap@n.
        total_num_positives : int
            You can specify the number of total positive in the list.
            If specified, it will be used in calculation.

        Returns
        --------
        ap: float
            The non-interpolated average precision at n. If n is larger than the
            length of the ranked list, the average precision will be returned.

        Raises
        -------
        ValueError:
            An error occurred when
            1) the format of the input is not the numpy 1-D array;
            2) the shape of predictions and labels does not match;
            3) the input n is not a positive integer.
        """

        def _shuffle(y_pred, y_true):
            random.seed(0)
            idx = random.sample(range(len(y_pred)), len(y_true))
            _predictions = y_pred[idx]
            _labels = y_true[idx]
            return _predictions, _labels

        if len(predictions) != len(labels):
            raise ValueError("the shape of predictions and labels does not match.")

        if n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be 'None' or a positive integer. It was '%s'." % n)

        # add a shuffler to avoid overestimating the ap
        predictions, labels = _shuffle(predictions, labels)
        sortidx = sorted(range(len(predictions)), key=lambda k: predictions[k], reverse=True)

        if total_num_positives is None:
            npos = np.size(np.where(labels > 0))
        else:
            npos = total_num_positives

        if npos == 0:
            return 0

        if n is not None:
            npos = min(npos, n)

        delta_recall = 1.0 / npos
        poscount = 0.0

        ap = 0.0

        # calculate the ap
        r = len(sortidx)
        if n is not None:
            r = min(r, n)
        for i in range(r):
            if labels[sortidx[i]] > 0:
                poscount += 1
                ap += poscount / (i + 1) * delta_recall

        return ap

    @staticmethod
    def _zero_one_normalize(predictions, epsilon=1e-7):
        """Normalize the predictions to the range between 0.0 and 1.0.

        For some predictions like SVM predictions, we need to normalize them before
        calculate the interpolated average precision. The normalization will not
        change the rank in the original list and thus won't change the average
        precision.

        Parameters
        -----------
        predictions: np.ndarray
            A numpy 1-D array storing the sparse prediction scores.
        epsilon: float
            A small constant to avoid denominator being zero.

        Returns
        --------
        ret: np.ndarray
            The normalized prediction.
        """

        denominator = np.max(predictions) - np.min(predictions)
        ret = (predictions - np.min(predictions)) / max(np.max(denominator), epsilon)
        return ret


class MeanAveragePrecisionCalculator(object):
    """Calculate the mean average precision. It provides an interface for calculating
    mean average precision for an entire list or the top-n ranked items.

    Example usages:
    We first call the function accumulate many times to process parts of the ranked
    list. After processing all the parts, we call peek_map_at_n
    to calculate the mean average precision.

    ```
    import random

    y_pred = np.array([[random.random() for _ in range(50)] for _ in range(1000)])
    y_true = np.array([[random.choice([0, 1]) for _ in range(50)] for _ in range(1000)])

    # mean average precision for 50 classes.
    calculator = MeanAveragePrecisionCalculator(num_class=50)
    calculator.store(y_pred, y_true)
    aps = calculator.peek_map_at_n()
    ```
    """

    def __init__(self, num_class, filter_empty_classes=True, top_n=None):
        """Construct a calculator to calculate the (macro) average precision.

        Parameters
        -----------
        num_class: int
            A positive Integer specifying the number of classes.
        top_n: int
            A positive Integer specifying the average precision at n, or None
            to use all provided data points.
        filter_empty_classes: bool
            Whether to filter classes without any positives.

        Raises
        -------
        ValueError:
            An error occurred when num_class or top_n is not a positive integer
        """

        if not isinstance(num_class, int) or num_class <= 1:
            raise ValueError("num_class must be a positive integer.")

        self._ap_calculators = []  # member of AveragePrecisionCalculator
        self._num_class = num_class  # total number of classes
        self._filter_empty_classes = filter_empty_classes
        for i in range(num_class):
            self._ap_calculators.append(AveragePrecisionCalculator(top_n))

    def store(self, predictions, labels, num_positives=None):
        """Accumulate the predictions and their ground truth labels.

        Parameters
        -----------
        predictions: list, np.ndarray
            A list of lists or np arrays of arrays storing the prediction scores.
        labels: list, np.ndarray
            A list of lists or np arrays of arrays storing the ground truth labels.
            Any value larger than 0 will be treated as positives, otherwise as negatives.
        num_positives: int, None
            If provided, it is a list of numbers representing the number of true positives for each class.
            If not provided, the number of true positives will be inferred from the 'labels' array.

        Raises
        -------
        ValueError:
            An error occurred when the shape of predictions and labels does not match.
        """

        if not num_positives:
            num_positives = [None for i in range(self._num_class)]

        calculators = self._ap_calculators
        for i in range(self._num_class):
            calculators[i].store(predictions[i], labels[i], num_positives[i])

    def clear(self):
        for calculator in self._ap_calculators:
            calculator.clear()

    def is_empty(self):
        return [calc.heap_size for calc in self._ap_calculators] == [0 for _ in range(self._num_class)]

    def peek_map_at_n(self):
        """Peek the non-interpolated mean average precision at n.

        Returns
        --------
        aps: list
            An array of non-interpolated average precision at n (default 0) for each class.
        """
        aps = []
        for i in range(self._num_class):
            if not self._filter_empty_classes or self._ap_calculators[i].num_accumulated_positives > 0:
                aps.append(self._ap_calculators[i].peek_ap_at_n())


        # aps = [self._ap_calculators[i].peek_ap_at_n() ]
        return aps
