#!/usr/bin/python3

from functools import partial
from datetime import datetime
import pandas as pd
from joblib import parallel_backend
import random
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
import shutil
import pathlib
import os
import math
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
import time
import copy
import random
import pickle
from joblib import Parallel, delayed
import tempfile
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
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

from pymrmre import mrmr
from pprint import pprint
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import mutual_info_classif


from mlflow import log_metric, log_param, log_artifact, log_dict, log_image


from loadData import *
from utils import *
from parameters import *
from extraFeatureSelections import *



### parameters
TrackingPath = "/data/results/radFS/mlrun.benchmark"



print ("Have", len(fselParameters["FeatureSelection"]["Methods"]), "Feature Selection Methods.")
print ("Have", len(clfParameters["Classification"]["Methods"]), "Classifiers.")


#    wie CV: alle parameter gehen einmal durch
def getExperiments (experimentList, expParameters, sKey, inject = None):
    newList = []
    for exp in experimentList:
        for cmb in list(itertools.product(*expParameters.values())):
            pcmb = dict(zip(expParameters.keys(), cmb))
            if inject is not None:
                pcmb.update(inject)
            _exp = exp.copy()
            _exp.append((sKey, pcmb))
            newList.append(_exp)
    experimentList = newList.copy()
    return experimentList



# this is pretty non-generic, maybe there is a better way, for now it works.
def generateAllExperiments (experimentParameters, verbose = False):
    experimentList = [ [] ]
    for k in experimentParameters.keys():
        if verbose == True:
            print ("Adding", k)
        if k == "BlockingStrategy":
            newList = []
            blk = experimentParameters[k].copy()
            newList.extend(getExperiments (experimentList, blk, k))
            experimentList = newList.copy()
        elif k == "FeatureSelection":
            # this is for each N too
            print ("Adding feature selection")
            newList = []
            for n in experimentParameters[k]["N"]:
                for m in experimentParameters[k]["Methods"]:
                    fmethod = experimentParameters[k]["Methods"][m].copy()
                    fmethod["nFeatures"] = [n]
                    newList.extend(getExperiments (experimentList, fmethod, m))
            experimentList = newList.copy()
        elif k == "Classification":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        else:
            experimentList = getExperiments (experimentList, experimentParameters[k], k)

    return experimentList



# if we do not want scaling to be performed on all data,
# we need to save thet scaler. same for imputer.
def preprocessData (X, y):
    simp = SimpleImputer(strategy="mean")
    X = pd.DataFrame(simp.fit_transform(X),columns = X.columns)

    sscal = StandardScaler()
    X = pd.DataFrame(sscal.fit_transform(X),columns = X.columns)
    return X, y


def applyFS (X, y, fExp):
    print ("Applying", fExp)
    return X, y



def applyCLF (X, y, cExp, fExp = None):
    print ("Training", cExp, "on FS:", fExp)
    return "model"



def testModel (y_pred, y_true, idx, fold = None):
    t = np.array(y_true)
    p = np.array(y_pred)

    # naive bayes can produce nan-- on ramella2018 it happens.
    # in that case we replace nans by 0
    p = np.nan_to_num(p)
    y_pred_int = [int(k>=0.5) for k in p]

    acc = accuracy_score(t, y_pred_int)
    df = pd.DataFrame ({"y_true": t, "y_pred": p}, index = idx)

    return {"y_pred": p, "y_test": t,
                "y_pred_int": y_pred_int,
                "idx": np.array(idx).tolist()}, df, acc



def getRunID (pDict):
    def dict_hash(dictionary: Dict[str, Any]) -> str:
        dhash = hashlib.md5()
        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    run_id = dict_hash(pDict)
    return run_id



def getAUCCurve (modelStats, dpi = 100):
    # compute roc and auc
    fpr, tpr, thresholds = roc_curve (modelStats["y_test"], modelStats["y_pred"])
    area_under_curve = auc (fpr, tpr)
    if (math.isnan(area_under_curve) == True):
        print ("ERROR: Unable to compute AUC of ROC curve. NaN detected!")
        print (modelStats["y_test"])
        print (modelStats["y_pred"])
        raise Exception ("Unable to compute AUC")
    sens, spec = findOptimalCutoff (fpr, tpr, thresholds)
    return area_under_curve, sens, spec



def getPRCurve (modelStats, dpi = 100):
    # compute roc and auc
    precision, recall, thresholds = precision_recall_curve(modelStats["y_test"], modelStats["y_pred"])
    try:
        f1 = f1_score (modelStats["y_test"], modelStats["y_pred_int"])
    except Exception as e:
        print (modelStats["y_test"])
        print (modelStats["y_pred_int"])
        raise (e)
    f1_auc = auc (recall, precision)
    if (math.isnan(f1_auc) == True):
        print ("ERROR: Unable to compute AUC of PR curve. NaN detected!")
        print (modelStats["y_test"])
        print (modelStats["y_pred"])
        raise Exception ("Unable to compute AUC")
    return f1, f1_auc



def logMetrics (foldStats):
    y_preds = []
    y_test = []
    y_index = []

    aucList = {}
    for k in foldStats:
        if "fold" in k:
            y_preds.extend(foldStats[k]["y_pred"])
            y_test.extend(foldStats[k]["y_test"])
            y_index.extend(foldStats[k]["idx"])
            fpr, tpr, thresholds = roc_curve (foldStats[k]["y_test"], foldStats[k]["y_pred"])
            area_under_curve = auc (fpr, tpr)
            aucList["AUC" + "_" + str(len(aucList))] = area_under_curve
    auc_mean = np.mean(list(aucList.values()))
    auc_std = np.std(list(aucList.values()))
    aucList["AUC_mean"] = auc_mean
    aucList["AUC_std"] = auc_std

    modelStats, df, acc = testModel (y_preds, y_test, idx = y_index, fold = "ALL")
    roc_auc, sens, spec = getAUCCurve (modelStats, dpi = 72)
    f1, f1_auc = getPRCurve (modelStats, dpi = 72)
    #pprint(aucList)
    log_dict(aucList, "aucStats.json")

    log_dict(modelStats, "params.yml")
    log_metric ("Accuracy", acc)
    log_metric ("Sens", sens)
    log_metric ("Spec", spec)
    log_metric ("AUC", roc_auc)
    log_metric ("F1", f1)
    log_metric ("F1_AUC", f1_auc)
    #print (foldStats["features"])
    log_dict(foldStats["features"], "features.json")
    for k in foldStats["params"]:
        log_param (k, foldStats["params"][k])
    with tempfile.TemporaryDirectory() as temp_dir:
        predFile = os.path.join(temp_dir, "preds.csv")
        df.to_csv(predFile)
        mlflow.log_artifact(predFile)
    print(".", end = '', flush=True)
    return {}



def createFSel (fExp, cache = True):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter=500, solver='liblinear', C = C)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures)


    if method == "ET":
        clf = ExtraTreesClassifier()
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures)


    if method == "ReliefF":
        from ITMO_FS.filters.univariate import reliefF_measure
        pipe = SelectKBest(reliefF_measure, k = nFeatures)


    if method == "MIM":
        pipe = SelectKBest(mutual_info_classif, k = nFeatures)


    if method == "Chi2":
        from ITMO_FS.filters.univariate import chi2_measure
        pipe = SelectKBest(chi2_measure, k = nFeatures)


    if method == "Anova":
        from ITMO_FS.filters.univariate import anova
        pipe = SelectKBest(anova, k = nFeatures)


    if method == "InformationGain":
        from ITMO_FS.filters.univariate import information_gain
        pipe = SelectKBest(information_gain, k = nFeatures)


    if method == "GiniIndex":
        from ITMO_FS.filters.univariate import gini_index
        pipe = SelectKBest(gini_index, k = nFeatures)


    if method == "SUMeasure":
        from ITMO_FS.filters.univariate import su_measure
        pipe = SelectKBest(su_measure, k = nFeatures)


    if method == "FCBF":
        from ITMO_FS.filters.multivariate.FCBF import FCBFDiscreteFilter
        def fcbf_fct (X, y):
            fcbf = FCBFDiscreteFilter()
            fcbf.fit(X,y)
            idxList = fcbf.selected_features
            scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
            return np.array(scores)
        pipe = SelectKBest(fcbf_fct, k = nFeatures)


    if method == "MCFS":
        from ITMO_FS.filters import MCFS
        def mcfs_fct (X, y):
            mcfs = MCFS(nFeatures, scheme='0-1') # dot is broken
            idxList = mcfs.feature_ranking(X)
            scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
            return np.array(scores)
        pipe = SelectKBest(mcfs_fct, k = nFeatures)


    if method == "UDFS":
        from ITMO_FS.filters import UDFS
        def udfs_fct (X, y):
            udfs = UDFS(nFeatures)
            idxList = udfs.feature_ranking(X)
            scores = [1 if idx in idxList else 0 for idx in range(X.shape[1])]
            return np.array(scores)
        pipe = SelectKBest(udfs_fct, k = nFeatures)


    if method == "Pearson":
        from ITMO_FS.filters.univariate import pearson_corr
        pipe = SelectKBest(pearson_corr, k = nFeatures)


    if method == "Kendall":
        from scipy.stats import kendalltau
        def kendall_corr_fct (X, y):
            scores = [0]*X.shape[1]
            for k in range(X.shape[1]):
                scores[k] = 1-kendalltau(X[:,k], y)[1]
            return np.array(scores)
        pipe = SelectKBest(kendall_corr_fct, k = nFeatures)


    if method == "Fechner":
        from ITMO_FS.filters.univariate import fechner_corr
        pipe = SelectKBest(fechner_corr, k = nFeatures)


    if method == "Spearman":
        from ITMO_FS.filters.univariate import spearman_corr
        pipe = SelectKBest(spearman_corr, k = nFeatures)


    if method == "Laplacian":
        from ITMO_FS.filters.univariate import laplacian_score
        def laplacian_score_fct (X, y):
            scores = laplacian_score(X,y)
            return -scores
        pipe = SelectKBest(laplacian_score_fct, k = nFeatures)


    if method == "FisherScore":
        from ITMO_FS.filters.univariate import f_ratio_measure
        pipe = SelectKBest(f_ratio_measure, k = nFeatures)


    if method == "Relief":
        from extraFeatureSelections import relief_measure
        pipe = SelectKBest(relief_measure, k = nFeatures)


    if method == "JMI":
        from skfeature.function.information_theoretical_based import JMI
        def jmi_score (X, y, nFeatures):
            sol, _, _ = JMI.jmi (X,y, n_selected_features = nFeatures)
            scores = [0]*X.shape[1]
            for j,z in enumerate(sol):
                scores[z] = (len(sol) - j)/len(sol)
            scores = np.asarray(scores, dtype = np.float32)
            return scores
        jmi_score_fct = partial(jmi_score, nFeatures = nFeatures)
        pipe = SelectKBest(jmi_score_fct, k = nFeatures)


    if method == "ICAP":
        from skfeature.function.information_theoretical_based import ICAP
        def icap_score (X, y, nFeatures):
            sol, _, _ =ICAP.icap (X,y, n_selected_features = nFeatures)
            scores = [0]*X.shape[1]
            for j,z in enumerate(sol):
                scores[z] = (len(sol) - j)/len(sol)
            scores = np.asarray(scores, dtype = np.float32)
            return scores
        icap_score_fct = partial(icap_score, nFeatures = nFeatures)
        pipe = SelectKBest(icap_score_fct, k = nFeatures)


    # not exported
    if method == "DCSF":
        from ITMO_FS.filters.multivariate import DCSF
        def dcsf_score_fct (X, y):
            selected_features = []
            other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
            scores = DCSF(np.array(selected_features), np.array(other_features), X, y)
            return scores
        pipe = SelectKBest(dcsf_score_fct, k = nFeatures)


    if method == "CIFE":
        from skfeature.function.information_theoretical_based import CIFE
        def cife_score (X, y, nFeatures):
            sol, _, _ = CIFE.cife (X,y, n_selected_features = nFeatures)
            scores = [0]*X.shape[1]
            for j,z in enumerate(sol):
                scores[z] = (len(sol) - j)/len(sol)
            scores = np.asarray(scores, dtype = np.float32)
            return scores
        cife_score_fct = partial(cife_score, nFeatures = nFeatures)
        pipe = SelectKBest(cife_score_fct, k = nFeatures)


    # should be the same as MIM
    if method == "MIFS":
        from ITMO_FS.filters.multivariate import MIFS
        def mifs_score_fct (X, y):
            selected_features = []
            other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
            scores = MIFS(np.array(selected_features), np.array(other_features), X, y, beta = 0.5)
            return scores
        pipe = SelectKBest(mifs_score_fct, k = nFeatures)


    if method == "CMIM":
        from skfeature.function.information_theoretical_based import CMIM
        def cmim_score (X, y, nFeatures):
            sol, _, _ =CMIM.cmim (X,y, n_selected_features = nFeatures)
            scores = [0]*X.shape[1]
            for j,z in enumerate(sol):
                scores[z] = (len(sol) - j)/len(sol)
            scores = np.asarray(scores, dtype = np.float32)
            return scores
        cmim_score_fct = partial(cmim_score, nFeatures = nFeatures)
        pipe = SelectKBest(cmim_score_fct, k = nFeatures)


    if method == "MRI":
        from ITMO_FS.filters.multivariate import MRI
        def mri_score_fct (X, y):
            selected_features = []
            other_features = [i for i in range(0, X.shape[1]) if i not in selected_features]
            scores = MRI(np.array(selected_features), np.array(other_features), X, y)
            return scores
        pipe = SelectKBest(mri_score_fct, k = nFeatures)


    if method == "MRMR":
        def mrmr_score (X, y, nFeatures):
            Xp = pd.DataFrame(X, columns = range(X.shape[1]))
            yp = pd.DataFrame(y, columns=['Target'])

            # we need to pre-specify the max solution length...
            solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=1)
            scores = [0]*Xp.shape[1]
            for j,z in enumerate(solutions.iloc[0][0]):
                scores[z] = (len(solutions.iloc[0][0]) - j)/len(solutions.iloc[0][0])
            scores = np.asarray(scores, dtype = np.float32)
            return scores
        mrmr_score_fct = partial(mrmr_score, nFeatures = nFeatures)
        pipe = SelectKBest(mrmr_score_fct, k = nFeatures)


    if method == "MRMRe":
        def mrmre_score (X, y, nFeatures):
            Xp = pd.DataFrame(X, columns = range(X.shape[1]))
            yp = pd.DataFrame(y, columns=['Target'])

            # we need to pre-specify the max solution length...
            solutions = mrmr.mrmr_ensemble(features = Xp, targets = yp, solution_length=nFeatures, solution_count=5)
            scores = [0]*Xp.shape[1]
            for k in solutions.iloc[0]:
                for j, z in enumerate(k):
                    scores[z] = scores[z] + Xp.shape[1] - j
            scores = np.asarray(scores, dtype = np.float32)
            scores = scores/np.sum(scores)
            return scores
        mrmre_score_fct = partial(mrmre_score, nFeatures = nFeatures)
        pipe = SelectKBest(mrmre_score_fct, k = nFeatures)


    if method == "SVMRFE":
        def svmrfe_score_fct (X, y):
            svc = LinearSVC (C=1)
            rfe = RFECV(estimator=svc, step=0.10, scoring='roc_auc', n_jobs=1)
            rfe.fit(X, y)
            scores = rfe.ranking_
            return scores
        pipe = SelectKBest(svmrfe_score_fct, k = nFeatures)


    if method == "Boruta":
        import boruta
        def boruta_fct (X, y):
            rfc = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample')
            b = boruta.BorutaPy (rfc, n_estimators = nFeatures)
            b.fit(X, y)
            scores = np.max(b.ranking_) - b.ranking_
            return scores
        pipe = SelectKBest(boruta_fct, k = nFeatures)


    if method == "RandomizedLR":
        from sklearn.utils import resample
        def randlr_fct (X, y):
            # only 100 instead of 1000
            scores = None
            for k in range(25):
                boot = resample(range(0,X.shape[0]), replace=True, n_samples=X.shape[0], random_state=k)
                model = LogisticRegression(solver = 'lbfgs', random_state = k)
                model.fit(X[boot,:], y[boot])
                if scores is None:
                    scores = model.coef_[0]*0
                scores = scores + np.abs(model.coef_[0])
            return scores
        pipe = SelectKBest(randlr_fct, k = nFeatures)


    if method == "tScore":
        from skfeature.function.statistical_based import t_score
        pipe = SelectKBest(t_score.t_score, k = nFeatures)


    if method == "Wilcoxon":
        from extraFeatureSelections import wilcoxon_score
        pipe = SelectKBest(wilcoxon_score, k = nFeatures)


    if method == "Variance":
        def variance (X, y):
            scores = np.var(X, axis = 0)
            return scores
        pipe = SelectKBest(variance, k = nFeatures)


    if method == "TraceRatio":
        from skfeature.function.similarity_based import trace_ratio
        def trace_ratio_score (X, y, nFeatures):
            fidx, fscore, _ = trace_ratio.trace_ratio (X,y, n_selected_features = nFeatures)
            scores = [0]*X.shape[1]
            for j in range(len(fidx)):
                scores[fidx[j]] = fscore[j]
            scores = np.asarray(scores, dtype = np.float32)
            return scores
        trace_ratio_score_fct = partial(trace_ratio_score, nFeatures = nFeatures)
        pipe = SelectKBest(trace_ratio_score_fct, k = nFeatures)


    if method == "Bhattacharyya":
        def bhattacharyya_score_fct (X, y):
            import cv2
            yn = y/np.sum(y)
            yn = np.asarray(yn, dtype = np.float32)
            scores = [0]*X.shape[1]
            for j in range(X.shape[1]):
                xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
                xn = xn/np.sum(xn)
                xn = np.asarray(xn, dtype = np.float32)
                scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

            scores = np.asarray(scores, dtype = np.float32)
            return -scores
        pipe = SelectKBest(bhattacharyya_score_fct, k = nFeatures)


    if method == "None":
        def dummy_score (X, y):
            scores = np.ones(X.shape[1])
            return scores
        pipe = SelectKBest(dummy_score, k = 'all')

    return pipe



def createClf (cExp):
    #print (cExp)
    method = cExp[0][0]
    if method == "Constant":
        model = DummyClassifier()

    if method == "SVM":
        C = cExp[0][1]["C"]
        svc = LinearSVC(C = C)
        model = CalibratedClassifierCV(svc)

    if method == "RBFSVM":
        C = cExp[0][1]["C"]
        g = cExp[0][1]["gamma"]
        model = SVC(kernel = "rbf", C = C, gamma = g, probability = True)

    if method == "LDA":
        model = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')

    if method == "QDA":
        model = QuadraticDiscriminantAnalysis()

    if method == "LogisticRegression":
        C = cExp[0][1]["C"]
        model = LogisticRegression(solver = 'lbfgs', C = C, random_state = 42)

    if method == "RandomForest":
        n_estimators = cExp[0][1]["n_estimators"]
        model = RandomForestClassifier(n_estimators = n_estimators)

    if method == "kNN":
        neighbors = cExp[0][1]["N"]
        model = KNeighborsClassifier(neighbors)

    if method == "XGBoost":
        learning_rate = cExp[0][1]["learning_rate"]
        n_estimators = cExp[0][1]["n_estimators"]
        model = XGBClassifier(learning_rate = learning_rate, n_estimators = n_estimators,  n_jobs = 1,  use_label_encoder=False, eval_metric = "logloss", random_state = 42)

    if method == "XGBoost_GPU":
        learning_rate = cExp[0][1]["learning_rate"]
        n_estimators = cExp[0][1]["n_estimators"]
        model = XGBClassifier(learning_rate = learning_rate, n_estimators = n_estimators, use_label_encoder=False, eval_metric = "logloss", tree_method='gpu_hist', random_state = 42)

    if method == "NaiveBayes":
        model = GaussianNB()

    if method == "NeuralNetwork":
        N1 = cExp[0][1]["layer_1"]
        N2 = cExp[0][1]["layer_2"]
        N3 = cExp[0][1]["layer_3"]
        model = MLPClassifier (hidden_layer_sizes=(N1,N2,N3,), random_state=42, max_iter = 1000)
    return model



@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
def executeExperiment (fselExperiments, clfExperiments, data, dataID):
    mlflow.set_tracking_uri(TrackingPath)

    y = data["Target"]
    X = data.drop(["Target"], axis = 1)
    X, y = preprocessData (X, y)

    # need a fixed set of folds to be comparable
    kfolds = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 1, random_state = 42)

    # make sure experiment gets selected
    raceOK = False
    while raceOK == False:
        try:
            mlflow.set_experiment(dataID)
            raceOK = True
        except:
            time.sleep(0.5)
            pass

    stats = {}
    for i, fExp in enumerate(fselExperiments):
        np.random.seed(i)
        random.seed(i)
        for j, cExp in enumerate(clfExperiments):
            timings = {}

            foldStats = {}
            foldStats["features"] = []
            foldStats["params"] = {}
            foldStats["params"].update(fExp)
            foldStats["params"].update(cExp)
            run_name = getRunID (foldStats["params"])

            current_experiment = dict(mlflow.get_experiment_by_name(dataID))
            experiment_id = current_experiment['experiment_id']

            # check if we have that already
            # recompute using mlflow did not work, so i do my own.
            if len(glob (os.path.join(TrackingPath, str(experiment_id), "*/artifacts/" + run_name + ".ID"))) > 0:
                print ("X", end = '', flush = True)
                continue

            # log what we do next
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("(RUN) " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")


            expVersion = '_'.join([k for k in foldStats["params"] if "Experiment" not in k])
            pID = str(foldStats["params"])

            # register run in mlflow now
            run_id = getRunID (foldStats["params"])
            mlflow.start_run(run_name = run_id, tags = {"Version": expVersion, "pID": pID})
            # this is stupid, but well, log a file with name=runid
            log_dict(foldStats["params"], run_id+".ID")

            for k, (train_index, test_index) in enumerate(kfolds.split(X, y)):
                X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
                y_train, y_test = y[train_index].copy(), y[test_index].copy()

                # log fold index too
                log_dict({"Test": test_index.tolist(), "Train": train_index.tolist()}, "CVIndex_"+str(k)+".json")

                fselector = createFSel (fExp)
                with np.errstate(divide='ignore',invalid='ignore'):
                    timeFSStart = time.time()
                    fselector.fit (X_train.copy(), y_train.copy())
                    timeFSEnd = time.time()
                    timings["Fsel_Time_Fold_" + str(k)] = timeFSEnd - timeFSStart
                feature_idx = fselector.get_support()
                selected_feature_names = X_train.columns[feature_idx].copy()
                all_feature_names = X_train.columns.copy()

                # log also 0-1
                fpat = np.zeros(X_train.shape[1])
                for j,f in enumerate(feature_idx):
                    fpat[j] = int(f)
                # just once
                if k == 0:
                    log_dict({f:fpat[j] for j, f in enumerate(all_feature_names)}, "FNames_"+str(k)+".json")
                log_dict({j:fpat[j] for j, f in enumerate(all_feature_names)}, "FPattern_"+str(k)+".json")
                foldStats["features"].append(list([selected_feature_names][0].values))

                # apply selector-- now the data is numpy, not pandas, lost its names
                X_fs_train = fselector.transform (X_train)
                y_fs_train = y_train

                X_fs_test = fselector.transform (X_test)
                y_fs_test = y_test

                # check if we have any features
                if X_fs_train.shape[1] > 0:
                    classifier = createClf (cExp)

                    timeClfStart = time.time()
                    classifier.fit (X_fs_train, y_fs_train)
                    timeClfEnd = time.time()
                    timings["Clf_Time_Fold_" + str(k)] = timeClfEnd - timeClfStart

                    y_pred = classifier.predict_proba (X_fs_test)
                    y_pred = y_pred[:,1]
                    foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)
                else:
                    # this is some kind of bug. if lasso does not select any feature and we have the constant
                    # classifier, then we cannot just put a zero there. else we get a different model than
                    # the constant predictor. we fix this by testing
                    if  cExp[0][0] == "Constant":
                        print ("F:", fExp, end = '')
                        classifier = createClf (cExp)
                        classifier.fit (X_train.iloc[:,0:2], y_train)
                        y_pred = classifier.predict_proba (X_fs_test)[:,1]
                    else:
                        # else we can just take 0 as a prediction
                        y_pred = y_test*0 + 1
                    foldStats["fold_"+str(k)], df, acc = testModel (y_pred, y_fs_test, idx = test_index, fold = k)

            stats[str(i)+"_"+str(j)] = logMetrics (foldStats)
            log_dict(timings, "timings.json")
            mlflow.end_run()
            with open(os.path.join(TrackingPath, "curExperiments.txt"), "a") as f:
                f.write("(DONE)" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + str(fExp) + "+" + str(cExp) + "\n")




def executeExperiments (z):
    fselExperiments, clfExperiments, data, d = z
    executeExperiment ([fselExperiments], [clfExperiments], data, d)



if __name__ == "__main__":
    print ("Hi.")

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # load data first
    datasets = {}
    dList = ["Li2020", "Carvalho2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Ramella2018",   "Keek2020", "Park2020", "Song2020" , "Toivonen2019"]
    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        print ("\tLoaded data with shape", datasets[d].shape)
        # avoid race conditions later
        try:
            mlflow.set_tracking_uri(TrackingPath)
            mlflow.create_experiment(d)
            mlflow.set_experiment(d)
            time.sleep(3)
        except:
            pass


    for d in dList:
        print ("\nExecuting", d)
        data = datasets[d]

        # generate all experiments
        fselExperiments = generateAllExperiments (fselParameters)
        print ("Created", len(fselExperiments), "feature selection parameter settings")
        clfExperiments = generateAllExperiments (clfParameters)
        print ("Created", len(clfExperiments), "classifier parameter settings")
        print ("Total", len(clfExperiments)*len(fselExperiments), "experiments")

        # generate list of experiment combinations
        clList = []
        for fe in fselExperiments:
            for clf in clfExperiments:
                clList.append( (fe, clf, data, d))

        # execute
        ncpus = 16
        with parallel_backend("loky", inner_max_num_threads=1):
            fv = Parallel (n_jobs = ncpus)(delayed(executeExperiments)(c) for c in clList)

#
