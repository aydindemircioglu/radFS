from collections import OrderedDict
import numpy as np


fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64],
        "Methods": {
            "None": {},
            "UDFS": {},
            "MCFS": {},
            "Fechner": {},
            #"FCBF": {},  # removed because of buggy results, see README
            "Boruta": {},
            "Laplacian": {},
            "CMIM": {},
            "MIM": {},
            #"CIFE": {}, # takes very long too
            "Relief": {},
            "Spearman": {},
            "MRMR": {},
            "LASSO": {"C": [1.0]},
            "SVMRFE": {},
            "ET": {},
            "FisherScore": {},
            "JMI": {},
            "Anova": {},
            "InformationGain": {},
            "SUMeasure": {},
            "Pearson": {},
            "GiniIndex": {},
            #"DCSF": {}, # removed because of buggy results, see README
            "Bhattacharyya": {},
            #"TraceRatio": {}, # definitely broken and kills memory or infinite loop
            "Variance" : {},
            "RandomizedLR": {},
            "Kendall": {},
            "Wilcoxon": {},
            "ICAP": {},
            "ReliefF": {},
            "MRMRe": {},
            "tScore": {},
        }
    }
})

clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "Constant": {},
            "LDA": {},
            "QDA": {},
            "kNN": {"N": [3, 5, 9]},
            "SVM": {"C":np.logspace(-6, 6, 7, base = 2.0)},
            "RBFSVM": {"C":np.logspace(-6, 6, 7, base = 2.0), "gamma":["auto"]},
            "RandomForest": {"n_estimators": [50, 250, 500]},
            "XGBoost": {"learning_rate": [0.001, 0.1, 0.3, 0.9], "n_estimators": [50, 250, 500]},
            "LogisticRegression": {"C": np.logspace(-6, 6, 7, base = 2.0) },
            "NeuralNetwork": {"layer_1": [4, 16, 64], "layer_2": [4, 16, 64], "layer_3": [4, 16, 64]},
            "NaiveBayes": {}
        }
    }
})


#
