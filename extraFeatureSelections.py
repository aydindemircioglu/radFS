from sklearn.metrics import roc_curve, auc, roc_auc_score
from math import sqrt
import numpy as np
import scipy.stats
from scipy import stats
import shutil
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances


# modified from t_score in skfeatures
def wilcoxon_score(X, y):
    """
    This function calculates t_score for each feature, where t_score is only used for binary problem
    t_score = |mean1-mean2|/sqrt(((std1^2)/n1)+((std2^2)/n2)))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}, shape (n_features,)
        t-score for each feature
    """

    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    c = np.unique(y)
    if len(c) == 2:
        for i in range(n_features):
            st, p = wilcoxon(X[:, i], y, zero_method='wilcox', correction=False, alternative='two-sided', mode='auto')
            F[i] = 1 - p
    else:
        print('y should be guaranteed to a binary class vector')
        exit(0)
    return np.abs(F)


# from https://github.com/ctlab/ITMO_FS/blob/5d1d0bfa66ab08fadf748ff115710b4faa03d5e4/ITMO_FS/utils/functions.py
def knn_from_class(distances, y, index, k, cl, anyOtherClass=False,
                   anyClass=False):
    """Return the indices of k nearest neighbors of X[index] from the selected
    class.
    Parameters
    ----------
    distances : array-like, shape (n_samples, n_samples)
        The distance matrix of the input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    index : int
        The index of an element.
    k : int
        The amount of nearest neighbors to return.
    cl : int
        The class label for the nearest neighbors.
    anyClass : bool
        If True, returns neighbors not belonging to the same class as X[index].
    Returns
    -------
    array-like, shape (k,) - the indices of the nearest neighbors
    """
    y_c = np.copy(y)
    if anyOtherClass:
        cl = y_c[index] + 1
        y_c[y_c != y_c[index]] = cl
    if anyClass:
        y_c.fill(cl)
    class_indices = np.nonzero(y_c == cl)[0]
    distances_class = distances[index][class_indices]
    nearest = np.argsort(distances_class)
    if y_c[index] == cl:
        nearest = nearest[1:]

    return class_indices[nearest[:k]]


# from https://github.com/ctlab/ITMO_FS/blob/5d1d0bfa66ab08fadf748ff115710b4faa03d5e4/ITMO_FS/filters/univariate/measures.py
def relief_measure(x, y, m=None, random_state=42):
    """Calculate Relief measure for each feature. This measure is supposed to
    work only with binary classification datasets; for multi-class problems use
    the ReliefF measure. Bigger values mean more important features.
    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    m : int, optional
        Amount of iterations to do. If not specified, n_samples iterations
        would be performed.
    random_state : int, optional
        Random state for numpy random.
    Returns
    -------
    array-like, shape (n_features,) : feature scores
    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and
    review. Journal of Biomedical Informatics 85 (2018) 189–203
    Examples
    --------
    >>> from ITMO_FS.filters.univariate import relief_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2])
    >>> relief_measure(x, y)
    array([ 0.    , -0.6   , -0.1875, -0.15  , -0.4   ])
    """
    weights = np.zeros(x.shape[1])
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 1:
        raise ValueError("Cannot calculate relief measure with 1 class")
    if 1 in counts:
        raise ValueError(
            "Cannot calculate relief measure because one of the classes has "
            "only 1 sample")

    n_samples = x.shape[0]
    n_features = x.shape[1]
    if m is None:
        m = n_samples

    x_normalized = MinMaxScaler().fit_transform(x)
    dm = euclidean_distances(x_normalized, x_normalized)
    indices = np.random.default_rng(random_state).integers(
        low=0, high=n_samples, size=m)
    objects = x_normalized[indices]
    hits_diffs = np.square(
        np.vectorize(
            lambda index: (
                x_normalized[index]
                - x_normalized[knn_from_class(dm, y, index, 1, y[index])]),
            signature='()->(n,m)')(indices))
    misses_diffs = np.square(
        np.vectorize(
            lambda index: (
                x_normalized[index]
                - x_normalized[knn_from_class(
                    dm, y, index, 1, y[index], anyOtherClass=True)]),
            signature='()->(n,m)')(indices))

    H = np.sum(hits_diffs, axis=(0,1))
    M = np.sum(misses_diffs, axis=(0,1))

    weights = M - H

    return weights / m


#
