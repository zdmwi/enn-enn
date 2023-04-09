import numpy as np

def xavier_initialization(size_in, size_out):
    """Implements Xavier or Glorot intitialization with the recommended biases"""
    interval = np.sqrt(6 / (size_in + size_out))
    weights = np.random.uniform(-interval, interval, (size_in, size_out))
    biases = np.zeros((1, size_out))
    return weights, biases


def kaiming_initialization(size_in, size_out):
    """Implements Kaiming or He initialization with the recommended biases"""
    weights = np.random.normal(0, np.sqrt(2 / size_in), (size_in, size_out))
    biases = np.zeros((1, size_out))
    return weights, biases


def get_batches(X, y, batch_size):
    """Returns a generator that yields batches of size batch_size"""
    N = X.shape[0]
    for i in range(0, N, batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size] if y is not None else None


def one_hot_encode(labels, n_categories):
    encoding = labels[:, None] == np.arange(n_categories)[None]
    return encoding.astype(np.float64)