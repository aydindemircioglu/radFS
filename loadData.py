import numpy as np
import os
import pandas as pd
from scipy.io import arff
import scipy.io as sio
from pprint import pprint
from sklearn.impute import SimpleImputer



# Define a class
class DataSet:
    def __init__(self, name):
        self.name = name

    def info (self):
        print("Dataset:", str(type(self).__name__), "\tDOI:", self.ID)

    def getData (self, folder):
        print("This octopus is " + self.color + ".")
        print(self.name + " is the octopus's name.")


class Song2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0237587"
        self.folder = None

    def getData (self, folder):
        self.folder = folder
        dataDir = os.path.join(folder, "journal.pone.0237587/")
        inputFile = "numeric_feature.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile), sep=",")
        data["Target"] = np.asarray(data["label"] > 0.5, dtype = np.uint8)
        data = data.drop(["Unnamed: 0", "label"], axis = 1)
        return (data)

    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "shape" in k or "info_V" in k]
        histoFeats = [k for k in data.keys() if "firstorder" in k]
        textureFeats = [k for k in data.keys() if k not in shapeFeats and k not in histoFeats]
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}



class Keek2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0232639"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0232639/Peritumoral-HN-Radiomics/")

        inputFile = "Clinical_DESIGN.csv"
        clDESIGNdata = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
        df = clDESIGNdata.copy()
        # remove those pats who did not die and have FU time less than 3 years
        df = clDESIGNdata[(clDESIGNdata["StatusDeath"].values == 1) | (clDESIGNdata["TimeToDeathOrLastFU"].values > 3*365)]
        target = df["TimeToDeathOrLastFU"] < 3*365
        target = np.asarray(target, dtype = np.uint8)

        inputFile = "Radiomics_DESIGN.csv"
        rDESIGNdata = pd.read_csv(os.path.join(dataDir, inputFile), sep=";")
        rDESIGNdata = rDESIGNdata.drop([z for z in rDESIGNdata.keys() if "General_" in z], axis = 1)
        rDESIGNdata = rDESIGNdata.loc[df.index]
        rDESIGNdata = rDESIGNdata.reset_index(drop = True)
        rDESIGNdata["Target"] = target

        # convert strings to float
        rDESIGNdata = rDESIGNdata.applymap(lambda x: float(str(x).replace(",", ".")))
        rDESIGNdata["Target"] = target

        return rDESIGNdata


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "Shape" in k]
        histoFeats = [k for k in data.keys() if "Stats" in k or "IH_" in k or "LocInt" in k]
        textureFeats = [k for k in data.keys() if "NGTDM" in k or  "NTLDM" in k or "NGLDM" in k or "GLCM" in k or "GLDZM" in k or "Fractal" in k or "GLRLM" in k or "GLSZM" in k]
        miss = [k for k in data.keys() if k not in textureFeats and k not in histoFeats and k not in shapeFeats]
        if len(miss) > 0:
            raise Exception ("Fix me")
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}



class Li2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0227703"

    def getData (self, folder):
        # clinical description not needed
        dataDir = os.path.join(folder, "journal.pone.0227703/")
        inputFile = "pone.0227703.s014.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        data["Target"] = data["Label"]
        data = data.drop(["Label"], axis = 1)
        return data


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        tKeys = ["AllDirection", "_angle", "Hara", "AngularSecondMoment", "Emphasis", "Zone", "Variability"]
        textureFeats = [k for t in tKeys for k in data.keys() if t in k]
        sKeys = ["VolumeCount", "VoxelValueSum", "Compactness", "Diameter", "Spheric", "Surface"]
        shapeFeats  = [k for t in sKeys for k in data.keys() if t in k]
        histoFeats  = [k for k in data.keys() if k not in textureFeats and k not in shapeFeats]
        # we took duplicates. maybe
        textureFeats = list(set(textureFeats))
        shapeFeats = list(set(shapeFeats))
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}




class Park2020 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0227315"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0227315/")
        inputFile = "pone.0227315.s003.xlsx"
        data = pd.read_excel(os.path.join(dataDir, inputFile), engine='openpyxl')
        target = data["pathological lateral LNM 0=no, 1=yes"]
        data = data.drop(["Patient No.", "pathological lateral LNM 0=no, 1=yes",
            "Sex 0=female, 1=male", "pathological central LNM 0=no, 1=yes"], axis = 1)
        data["Target"] = target
        return data


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "Shape" in k]
        shapeFeats
        hKeys = ["_"+str(k) for k in range(15)]
        histoFeats  = [t for k in hKeys for t in data.keys() if t.endswith(k)]

        tKeys = ["_"+str(k) for k in range(23,56)]
        textureFeats = [t for k in tKeys for t in data.keys() if t.endswith(k)]
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}



class Toivonen2019 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0217702"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0217702/")
        inputFile = "lesion_radiomics.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        data["Target"] = np.asarray(data["gleason_group"] > 0.0, dtype = np.uint8)
        data = data.drop(["gleason_group", "id"], axis = 1)
        return data


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = []
        hKeys = ["all-stats"]
        histoFeats  = [k for t in hKeys for k in data.keys() if t in k]
        tKeys = ["glcm", "lbp",  "zernike", "gabor", "haar", "-hu", "-hog", "sobel"]
        textureFeats = [k for t in tKeys for k in data.keys() if t in k]
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}





class Hosny2018A (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "HarvardRT.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        return data


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "shape" in k or "info_V" in k]
        histoFeats = [k for k in data.keys() if "firstorder" in k]
        textureFeats = [k for k in data.keys() if k not in shapeFeats and k not in histoFeats]
        miss = [k for k in data.keys() if k not in textureFeats and k not in histoFeats and k not in shapeFeats]
        if len(miss) > 0:
            raise Exception ("Fix me")
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}



class Hosny2018B (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "Maastro.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        return data


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "shape" in k or "info_V" in k]
        histoFeats = [k for k in data.keys() if "firstorder" in k]
        textureFeats = [k for k in data.keys() if k not in shapeFeats and k not in histoFeats]
        miss = [k for k in data.keys() if k not in textureFeats and k not in histoFeats and k not in shapeFeats]
        if len(miss) > 0:
            raise Exception ("Fix me")
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}



class Hosny2018C (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pmed.1002711"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pmed.1002711/" + "deep-prognosis/data/")
        # take only HarvardRT
        data = pd.read_csv(os.path.join(dataDir, "Moffitt.csv"))
        data = data.drop([z for z in data.keys() if "general_" in z], axis = 1)
        data["Target"] = data['surv2yr']
        # logit_0/logit_1 are possibly output of the CNN network
        data = data.drop(['id', 'surv2yr', 'logit_0', 'logit_1'], axis = 1)
        return data

    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "shape" in k or "info_V" in k]
        histoFeats = [k for k in data.keys() if "firstorder" in k]
        textureFeats = [k for k in data.keys() if k not in shapeFeats and k not in histoFeats]
        miss = [k for k in data.keys() if k not in textureFeats and k not in histoFeats and k not in shapeFeats]
        if len(miss) > 0:
            raise Exception ("Fix me")
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}





class Ramella2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0207455"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0207455/")
        inputFile = "pone.0207455.s001.arff"

        data = arff.loadarff(os.path.join(dataDir, inputFile))
        data = pd.DataFrame(data[0])
        data["Target"] = np.asarray(data['adaptive'], dtype = np.uint8)
        data = data.drop(['sesso', 'fumo', 'anni', 'T', 'N', "stadio", "istologia", "mutazione_EGFR", "mutazione_ALK", "adaptive"], axis = 1)
        return data

    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = []
        hKeys = ["all-stats"]
        histoFeats  = [k for t in hKeys for k in data.keys() if t in k]
        tKeys = ["LBP", "-1-", "101", "11", "10-", "100", "110", "1-1", "_0"]
        textureFeats = [k for t in tKeys for k in data.keys() if t in k]
        histoFeats  = [k for k in data.keys() if k not in textureFeats]
        # we took duplicates. maybe
        textureFeats = list(set(textureFeats))
        histoFeats = list(set(histoFeats))
        # numbers fit exactly the paper
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}





class Carvalho2018 (DataSet):
    def __init__(self):
        self.ID = "data:10.1371/journal.pone.0192859"

    def getData (self, folder):
        dataDir = os.path.join(folder, "journal.pone.0192859/")
        inputFile = "Radiomics.PET.features.csv"
        data = pd.read_csv(os.path.join(dataDir, inputFile))
        # all patients that are lost to followup were at least followed for two
        # years. that means if we just binarize the followup time using two years
        # we get those who died or did not die within 2 years as binary label
        data["Target"] = (data["Survival"] < 2.0)*1
        data = data.drop(["Survival", "Status"], axis = 1)
        return data


    def getFeatureTypes (self, folder = None):
        if folder is None:
            folder = self.folder
        # load and remove target
        data = self.getData (folder)
        data = data.drop(["Target"], axis = 1)

        shapeFeats = [k for k in data.keys() if "Shape" in k or "Volume" in k]
        histoFeats = [k for k in data.keys() if "Stats" in k or ("IVH" in k and "TLGRI" not in k)]
        textureFeats = [k for k in data.keys() if k not in shapeFeats and k not in histoFeats]
        return {"Shape": shapeFeats, "Histogram": histoFeats, "Texture": textureFeats}




if __name__ == "__main__":
    print ("Hi.")

    # obtain data sets
    datasets = {}
    dList = [ "Carvalho2018", "Hosny2018A", "Hosny2018B", "Hosny2018C", "Ramella2018",   "Toivonen2019", "Keek2020", "Li2020", "Park2020", "Song2020" ]

    for d in dList:
        eval (d+"().info()")
        datasets[d] = eval (d+"().getData('./data/')")
        df = datasets[d]

    # stats/ just a test
    for d in datasets:
        dimy = datasets[d].shape[0]/datasets[d].shape[1]
        b = np.round(100*(np.sum(datasets[d]["Target"])/len(datasets[d]) ))
        print (d, datasets[d].shape, dimy, b)
        print ("NAN:", datasets[d].isna().sum().sum())

    # test/display feature things
    for d in dList:
        print (d)
        featDict = eval (d+"().getFeatureTypes('./data/')")
        if datasets[d].shape[1]-1 != np.sum([len(featDict[k]) for k in featDict]):
            print ("### ERROR: Missing features:")
            print ("\t\tShould be:", datasets[d].shape[1])
            print ("\t\tBut is:", np.sum([len(featDict[k]) for k in featDict]))
        for k in featDict:
            print ("\t-", k, len(featDict[k]))
        sorted(featDict["Texture"])



#
