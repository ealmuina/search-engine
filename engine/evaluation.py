import numpy as np


def e_measure(relevant, retrieved, beta):
    pre = precision(relevant, retrieved)
    rec = recall(relevant, retrieved)
    den = beta ** 2 * (pre + rec)
    return (((1 + beta ** 2) * pre * rec) / den) if den else 0


def f_measure(relevant, retrieved):
    pre = precision(relevant, retrieved)
    rec = recall(relevant, retrieved)
    den = pre + rec
    return ((2 * pre * rec) / den) if den else 0


def precision(relevant, retrieved):
    ra = np.sum(relevant[retrieved]) if retrieved.shape[0] > 0 else 0
    a = retrieved.shape[0]
    return ra / a if a else 0


def recall(relevant, retrieved):
    ra = np.sum(relevant[retrieved]) if retrieved.shape[0] > 0 else 0
    r = np.sum(relevant)
    return ra / r


def r_precision(relevant, retrieved):
    r = np.sum(relevant)
    return precision(relevant, retrieved[:r])
