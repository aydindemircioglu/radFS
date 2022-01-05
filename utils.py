from sklearn.metrics import roc_curve, auc, roc_auc_score
from math import sqrt
import numpy as np
import scipy.stats
from scipy import stats
import shutil




def recreatePath (path, create = True):
        print ("Recreating path ", path)
        try:
                shutil.rmtree (path)
        except:
                pass

        if create == True:
            try:
                    os.makedirs (path)
            except:
                    pass
        print ("Done.")



def getOptimalThreshold (fpr, tpr, threshold, verbose = False):
    # own way
    minDistance = 2
    bestPoint = (2,-1)
    bestThreshold = 0
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p
            bestThreshold = i

    return thres[bestThreshold]




def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity


if __name__ == "__main__":
    y = np.asarray([1,0,0,0,1, 1,0,1,1,1, 1,0])
    A = np.asarray([0.4, 0.1, 0.2, 0.4, 0.2, 0.9, 0.7, 0.4, 0.1, 0.2, 0.9, 0.7])
    B = np.asarray([0.4, 0.3, 0.1, 0.2, 0.4, 0.2, 0.9, 0.7, 0.4, 0.1, 0.2, 0.4])
    p = testAUC (y, A, B)

#
