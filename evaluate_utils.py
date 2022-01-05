#!/usr/bin/python3

import json
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from scipy.stats import spearmanr
import pylab
import scipy.cluster.hierarchy as sch


from scipy.stats import pearsonr, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType


from docx import Document
from docx.shared import Inches

from datetime import datetime
import pandas as pd
import random
import numpy as np
import time
#import pycm
import shutil
import pathlib
import os
import math
import sys
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time
import copy
import random
import pickle
import tempfile
import itertools
import multiprocessing
import socket
from glob import glob
from collections import OrderedDict
import logging
import mlflow
from typing import Dict, Any
import hashlib
import json

from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from loadData import *
from utils import *
from parameters import *



### parameters
TrackingPath = "/data/results/radFS/mlrun.benchmark"
nCV = 10
DPI = 300



# https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#111111", edgecolor=['none'], alpha =0.15)
    return ax


# Modeling with Numpy
def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b)


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



def getResults (dList):
    if os.path.exists("./results/results.feather") == False:
        results = []
        for d in dList:
            current_experiment = dict(mlflow.get_experiment_by_name(d))
            experiment_id = current_experiment['experiment_id']
            runs = MlflowClient().search_runs(experiment_ids=experiment_id, max_results=50000)
            for r in runs:
                row = r.data.metrics
                row["UUID"] = r.info.run_uuid
                row["Model"] = r.data.tags["Version"]
                row["Parameter"] = r.data.tags["pID"]

                # stupid naming error
                row["Parameter"] = row["Parameter"]
                row["Model"] = row["Model"]

                row["FSel"], row["Clf"] = row["Model"].split("_")
                row["Dataset"] = d

                row["nFeatures"] = eval(row["Parameter"])[row["FSel"]]["nFeatures"]

                row["Path"] = os.path.join(TrackingPath,  str(experiment_id), str(r.info.run_uuid), "artifacts")
                results.append(row)

                # read timings
                apath = os.path.join(row["Path"], "timings.json")
                with open(apath) as f:
                    expData = json.load(f)
                row.update(expData)

                # read AUCs
                apath = os.path.join(row["Path"], "aucStats.json")
                with open(apath) as f:
                    aucData = json.load(f)
                row.update(aucData)

        results = pd.DataFrame(results)
        print ("Pickling results")
        pickle.dump (results, open("./results/results.feather","wb"))
    else:
        print ("Restoring results")
        results = pickle.load(open("./results/results.feather", "rb"))

    # AFTER EXPERIMENT: remove broken feature selection methods
    results = results.query("FSel != 'DCSF'").copy()
    results = results.query("FSel != 'FCBF'").copy()
    return results




def plotDendro (document, zMat, cMat, d = None, idx1 = None, fType = "All", sTitle = 'FIXME', filePrefix = "____"):
    D = zMat.copy()
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    fig = pylab.figure(figsize=(15,16), dpi = 300)
    if idx1 is None:
        Y = sch.linkage(cMat, method='single')
        Z2 = sch.dendrogram(Y)
        idx1 = Z2['leaves']
    D = D.iloc[idx1]
    D = D[D.keys()[idx1]]

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    im = axmatrix.matshow(D, norm=MidpointNormalize(midpoint=0,vmin=-1.0, vmax=1.0),\
                aspect='auto', origin='lower', cmap=pylab.cm.RdBu)
    axmatrix.set_yticks([])
    axmatrix.xaxis.tick_bottom()
    xaxis = np.arange(len(D.keys()))
    axmatrix.set_xticks(xaxis)
    axmatrix.set_yticks(xaxis)
    axmatrix.set_xticklabels(D.keys(), rotation = 90,   fontsize = 24)
    axmatrix.set_yticklabels(D.keys(), ha = "right", fontsize = 24)
    plt.rcParams['axes.titlepad'] = 0
    axmatrix.set_title (sTitle.format(d = d),  fontdict = {'fontsize' : 34})

    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    cbar = pylab.colorbar(im, cax=axcolor)
    cbar.ax.tick_params(labelsize=22)
    if d is None:
        fig.savefig(filePrefix + fType + '.png', bbox_inches='tight')
    else:
        fig.savefig(filePrefix + d + "_" + fType + '.png', bbox_inches='tight')
    return idx1




if __name__ == "__main__":
    print ("Hi.")

    # datasets
    dList = [ "Carvalho2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Ramella2018",   "Toivonen2019",
        "Keek2020", "Li2020", "Park2020", "Song2020" , ]

    results = getResults(dList)
#
