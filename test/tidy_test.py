from plotting import plot_histogram_with_ratio
from normalising_flow import NormalisingFlow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import os
import copy

normflow = NormalisingFlow()
url = "https://raw.githubusercontent.com/gputtley/ML-Assessment-3/master/tt_ggH.pkl"
data = pd.read_pickle(url)
data = data.loc[:,["pt_1","pt_2","m_vis","svfit_mass","pt_tt","eta_1","eta_2","dR","dphi"]]
#data = data.loc[:,["pt_1","pt_2","m_vis","svfit_mass","eta_1","eta_2"]]
normflow.data_loader.full = copy.deepcopy(data)
print(len(data))

normflow.batch_size = 2**9
normflow.epochs = 100
normflow.hidden_layers = [64]
normflow.num_layers = 16
normflow.flow_type = "RealNVP"

normflow.Train()

synth = normflow.GenerateData(10000)

num_bins = 40
ignore_quantile = 0.01
for i in data.columns:
  bin_edges = np.linspace(data.loc[:,i].quantile(ignore_quantile), data.loc[:,i].quantile(1-ignore_quantile), num_bins + 1)
  nom_hist,_ = np.histogram(data.loc[:,i],bins=bin_edges)
  samp_hist,be = np.histogram(synth.loc[:,i],bins=bin_edges)

  nom_hist = nom_hist.astype(float)
  samp_hist = samp_hist.astype(float)

  nom_hist_err = np.sqrt(nom_hist)
  samp_hist_err = np.sqrt(samp_hist)

  nom_norm = nom_hist.sum()
  samp_norm = samp_hist.sum()

  nom_hist *= 1/nom_norm
  samp_hist *= 1/samp_norm
  nom_hist_err *= 1/nom_norm
  samp_hist_err *= 1/samp_norm

  ratio = np.divide(samp_hist,nom_hist)
  plot_histogram_with_ratio(nom_hist,samp_hist,be,name_1="MC",name_2="Synthetic",xlabel=i,name="plots/synth_vs_data_"+i,errors_1=nom_hist_err,errors_2=samp_hist_err)

# try and separate datasets

# Create labels for the datasets (0 for nom_hist and 1 for samp_hist)
labels_nom = np.zeros(len(data.to_numpy()))
labels_samp = np.ones(len(synth.to_numpy()))

# Combine the datasets and labels
X = np.vstack((data.to_numpy(), synth.to_numpy()))
y = np.hstack((labels_nom, labels_samp))

inf_entries = np.isinf(synth.to_numpy())
# Check for values that are too large for float32
large_entries = np.abs(synth.to_numpy()) > np.finfo(np.float32).max

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Binary Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("ROC =",roc_auc)
