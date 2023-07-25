import sys
import os
from plotting import plot_histogram_with_ratio, plot_loss
import torch
import numpy as np
import pandas as pd
import normflows as nf
import matplotlib.pyplot as plt
from tqdm import tqdm
import mplhep as hep
import copy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def robust_scaling(dataframe):
    # Calculate the median and IQR of the data
    median = dataframe.median()
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1

    # Robust scaling
    scaled_data = (dataframe - median) / iqr
    return scaled_data, median, iqr

def inverse_robust_scaling(scaled_data, median, iqr):
    # Inverse robust scaling
    original_data = scaled_data * iqr + median
    return original_data

# Extract the values from the DataFrame
url = "https://raw.githubusercontent.com/gputtley/ML-Assessment-3/master/tt_ggH.pkl"
non_scaled_df = pd.read_pickle(url)


non_scaled_df = non_scaled_df.loc[:,["pt_1","pt_2","m_vis","svfit_mass","pt_tt","eta_1","eta_2","dR","dphi"]]
#mean = non_scaled_df.mean()
#std = non_scaled_df.std()
#df = (non_scaled_df - mean) / std
df, median, iqr = robust_scaling(non_scaled_df)

test_size = 0.5
train_df, test_df = train_test_split(df, test_size=test_size)

# get info for training
target_data = torch.tensor(train_df.values,dtype=torch.float32)
test_data = torch.tensor(test_df.values,dtype=torch.float32)
ndim = len(df.columns)

# Set up model
base = nf.distributions.base.DiagGaussian(ndim)

# Define list of flows
num_layers = 16
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(ndim)])
flows = []
for i in range(num_layers):
    s = nf.nets.MLP([ndim, 128, ndim], init_zeros=True)
    t = nf.nets.MLP([ndim, 128, ndim], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(ndim)]


# Construct flow model
model = nf.NormalizingFlow(base, flows)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

# Train model
batch_size = 256
epochs = 1000
max_iter = int(epochs * np.ceil(len(target_data) / batch_size))
max_iter_per_epoch = max_iter/epochs

train_loss_hist = np.array([])
test_loss_hist = np.array([])

# Define the maximum gradient norm for clipping
max_grad_norm = 100.0

# early stopping
patience = 40  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in validation loss to be considered as improvement
best_model_state = model.state_dict()
best_test_loss = float('inf')
epochs_without_improvement = 0

# Define the L2 regularization strength
l2_regularization_strength = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=l2_regularization_strength)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    num_rows = target_data.size(0)
    random_indices = torch.randperm(num_rows)[:batch_size]
    x = target_data[random_indices]

    # Compute train loss
    train_loss = model.forward_kld(x)

    # Add L2 regularization term to the train loss
    l2_regularization = 0.0
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2)  # L2 norm of the parameter
    train_loss += l2_regularization_strength * l2_regularization


    # Do backprop and optimizer step
    if ~(torch.isnan(train_loss) | torch.isinf(train_loss)):
        train_loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

    if it % max_iter_per_epoch == 0:
        #print("Iteration:",it,"Loss:",train_loss.to('cpu').data.numpy())

        # Log train loss
        train_loss_hist = np.append(train_loss_hist, train_loss.to('cpu').data.numpy())

        # Compute test loss
        num_rows = test_data.size(0)
        random_indices = torch.randperm(num_rows)[:batch_size]
        x_test = test_data[random_indices]

        with torch.no_grad():
            test_loss = model.forward_kld(x_test)
            test_loss += l2_regularization_strength * l2_regularization
        test_loss_hist = np.append(test_loss_hist, test_loss.to('cpu').data.numpy())

        # Plot loss
        plot_loss(train_loss_hist,test_loss_hist,xlabel="Epoch")

        # Check if test loss has improved
        if test_loss + min_delta < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print("Early stopping! No improvement in test loss for", patience, "epochs.")
            break


# After early stopping, load the best model state back into the model
model.load_state_dict(best_model_state)

# Plot 1D distributions

# sample from the gaussian
nsamps = 10000
samp, log_prob = model.sample(nsamps)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

#non_scaled_samp = (pd.DataFrame(samp.detach().numpy(), columns=list(non_scaled_df.columns)) * std) + mean
non_scaled_samp = inverse_robust_scaling(pd.DataFrame(samp.detach().numpy(), columns=list(non_scaled_df.columns)), median, iqr)
non_scaled_samp = non_scaled_samp[~non_scaled_samp.isin([np.inf,-np.inf, np.nan]).any(axis=1)]

num_bins = 40
ignore_quantile = 0.01
for i in df.columns:
  bin_edges = np.linspace(non_scaled_df.loc[:,i].quantile(ignore_quantile), non_scaled_df.loc[:,i].quantile(1-ignore_quantile), num_bins + 1)
  nom_hist,_ = np.histogram(non_scaled_df.loc[:,i],bins=bin_edges)
  samp_hist,be = np.histogram(non_scaled_samp.loc[:,i],bins=bin_edges)

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
labels_nom = np.zeros(len(non_scaled_df.to_numpy()))
labels_samp = np.ones(len(non_scaled_samp.to_numpy()))

# Combine the datasets and labels
X = np.vstack((non_scaled_df.to_numpy(), non_scaled_samp.to_numpy()))
y = np.hstack((labels_nom, labels_samp))

inf_entries = np.isinf(non_scaled_samp.to_numpy())
# Check for values that are too large for float32
large_entries = np.abs(non_scaled_samp.to_numpy()) > np.finfo(np.float32).max

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Binary Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("ROC =",roc_auc)
