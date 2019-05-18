from tensorflow.python.keras.metrics import top_k_categorical_accuracy


def top5_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 5)


def top1_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, 1)