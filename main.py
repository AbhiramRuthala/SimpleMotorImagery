#import the necessary files
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne.decoding as CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation as WSE

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

#instantiate the dataset
dataset = BNCI2014_001()
dataset.subject_list = [1, 2, 3]

sessions = dataset.get_data(subjects=[1])

#consider the subject and their data.
subject = 1
session = "0train"
run = "0"
raw = sessions[subject][session][run]

#select the paradigm
print(dataset.paradigm)

paradigm = LeftRightImagery()

print(paradigm.datasets)

#X is a numpy array and meta consists of all the data regarding the subject and the session.
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

#create the pipeline. Use CSP to improve the signal-to-noise ratio on the EEG and use LDA to classify the EEG data.
pipeline = make_pipeline(CSP(n_components=8), LDA())

#call the evaluation class. Use WSE (WithinSessionEvaluation) to look at the training and testing partitions of data for one session. If you want to create a cache file to store the data, set hdf5_path to the intended file path. If you don't want the class to recalculate the scores every time you run the function, set overwrite = False.
evaluation = WSE(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=True,
    hdf5_path=None,
)

results = evaluation.process({"csp+lda": pipeline})

#Generate the graph to visualize the results of the session reading. This uses the seaborn package. Fig, ax and subplots are under matplotlib
fig, ax = plt.subplots(figsize=(8, 7))
results["subj"] = results["subject"].apply(str)
sns.barplot(x="score", y="subj", hue="session", data=results, orient="h", palette="viridis", ax=ax)
plt.show()
