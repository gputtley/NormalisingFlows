from plotting import plot_histogram_with_ratio
from normalising_flow import NormalisingFlow
import numpy as np
import pandas as pd

# Setup flow
loc = "data/ggHToTauTauToTauhTauh.parquet"
columns_to_use = ["pt_1","pt_2","m_vis","svfit_mass","pt_tt","eta_1","eta_2","dR","dphi"]
normflow = NormalisingFlow(loc,columns=columns_to_use)

# Set hyperparameters
hyperparameters = {
    "patience" : 5,
    "min_delta" : 0.001,
    "l2_regularisation_strength" : 0.01,
    "max_grad_norm" : 100.0,
    "batch_size" : 2**(9),
    "epochs" : 100,
    "learning_rate" : 5e-3,
    "step_size" : 4,
    "gamma" : 0.2,
    "hidden_layers" : [64],
    "num_layers" : 16,
}
normflow.SetParameters(hyperparameters)

# Train flow

normflow.Train()

# Save model

normflow.SaveModel("models/ggHToTauTauToTauhTauh.pth")

# Generate synthetic data

synth,prob = normflow.GenerateData(10000)

# Draw 1D distributions

bin_edges, nom_hists = normflow.data_loader.MakeHistograms(num_bins=40,ignore_quantile=0.01)
synth_hists = {k:np.histogram(pd.DataFrame(synth.detach().cpu().numpy(),columns=columns_to_use).loc[:,k],bins=v)[0] for k, v in bin_edges.items()}

for k, v in bin_edges.items():
        
  nom_hist = nom_hists[k].astype(float)
  synth_hist = synth_hists[k].astype(float)

  nom_hist_err = np.sqrt(nom_hist)
  synth_hist_err = np.sqrt(synth_hist)

  nom_norm = nom_hist.sum()
  synth_norm = synth_hist.sum()

  nom_hist *= 1/nom_norm
  synth_hist *= 1/synth_norm
  nom_hist_err *= 1/nom_norm
  synth_hist_err *= 1/synth_norm

  ratio = np.divide(synth_hist,nom_hist)
  plot_histogram_with_ratio(nom_hist,synth_hist,v,name_1="MC",name_2="Synthetic",xlabel=k,name="plots/synth_vs_data_"+k,errors_1=nom_hist_err,errors_2=synth_hist_err)
