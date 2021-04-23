import logging
import numpy as np

# Logger
def get_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger("darts")
    log_format = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


# Top k accuracy
def top_n_accuracy(preds, ts, n):
    """ 
    ts: np array (nb_observations,)
    preds: prediction probabilities np array (nb_observations, n_classes)
    """
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]
