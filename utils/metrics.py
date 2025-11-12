__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import numpy as np

def dice(pred, truth):
    num = 2 * (np.sum((pred * truth), axis=(1,2,3)))
    den = (np.sum(pred,axis=(1,2,3)) + np.sum(truth,axis=(1,2,3)))
    return num / den

def fpr(pred,truth):
    fp = np.sum(np.logical_and(pred == 1, truth == 0))
    tn = np.sum(np.logical_and(pred == 0, truth == 0))
    return fp/(fp+tn)
