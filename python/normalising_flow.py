import torch
import torch.optim.lr_scheduler as lr_scheduler
import normflows as nf
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
from plotting import plot_loss
from data_loader import DataLoader

class NormalisingFlow:

  def __init__(self):

    self.flow_type = "RealNVP"
    ### Hyperparameters
    # early stopping
    self.patience = 5  # Number of epochs to wait for improvement
    self.min_delta = 0.001  # Minimum change in validation loss to be considered as improvement
    # L2 regularisation strength
    self.l2_regularisation_strength = 0.01
    # Maximum gradient norm for clipping
    self.max_grad_norm = 100.0
    # Training parameters
    self.batch_size = 2**(12)
    self.epochs = 1000    
    self.learning_rate = 5e-3
    # scheduler
    self.step_size = 4
    self.gamma = 0.2

    ### Architecture
    self.hidden_layers = [128] # Number of hidden layers in each coupling layer
    self.num_layers = 16 # Number of coupling layers

    ### Dataset - this is temporary due to bad data loader
    self.data_loader = DataLoader()
   
    ### Other
    self.ndim = 1
    self.base = nf.distributions.base.DiagGaussian(self.ndim) 
    self.model = None
    self.optimiser = None
    self.best_model_state = None
    self.best_test_loss = float('inf')
    self.epochs_without_improvement = 0
    self.dataset_size = 1
    self.max_iter = int(self.epochs * np.floor(self.dataset_size / self.batch_size))
    self.max_iter_per_epoch = self.max_iter/self.epochs
    self.train_loss_hist = np.array([])
    self.test_loss_hist = np.array([])
    self.early_stopping = False
    self.scheduler = None
    self.random_seed = 42
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)


  def UpdateParameters(self):

    self.base = nf.distributions.base.DiagGaussian(self.ndim)
    self.GetModel()
    self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularisation_strength)
    self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=self.step_size, gamma=self.gamma)
    self.best_model_state = self.model.state_dict()
    self.max_iter = int(self.epochs * np.ceil(self.dataset_size / self.batch_size))
    self.max_iter_per_epoch = self.max_iter/self.epochs
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)

  def GetModel(self):

    flows = []
    if self.flow_type == "RealNVP":
      b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.ndim)])
      for i in range(self.num_layers):
        s = nf.nets.MLP([self.ndim] + self.hidden_layers + [self.ndim], init_zeros=True)
        t = nf.nets.MLP([self.ndim] + self.hidden_layers + [self.ndim], init_zeros=True)
        if i % 2 == 0:
          flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
          flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(self.ndim)]
    elif self.flow_type == 'Planar':
      for i in range(self.num_layers):
        flows.append(nf.flows.Planar((self.ndim,)))
    elif self.flow_type == 'Radial':
      for i in range(self.num_layers):
        flows.append(nf.flows.Radial((self.ndim,)))

    # Construct flow model
    self.model = nf.NormalizingFlow(self.base, flows)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    self.model = self.model.to(device)

  def TrainIteration(self,x,x_test,new_epoch=False):

    self.optimiser.zero_grad()

    # Compute train loss
    train_loss = self.model.forward_kld(x)

    # Add L2 regularisation term to the train loss
    l2_regularisation = 0.0
    for param in self.model.parameters():
        l2_regularisation += torch.norm(param, p=2)  # L2 norm of the parameter
    train_loss += self.l2_regularisation_strength * l2_regularisation


    # Do backprop and optimiser step
    if ~(torch.isnan(train_loss) | torch.isinf(train_loss)):
        train_loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimiser.step()

    # Reporting and early stopping
    if new_epoch:

        # Update learning rate
        self.scheduler.step()

        # Log train loss
        self.train_loss_hist = np.append(self.train_loss_hist, train_loss.to('cpu').data.numpy())

        with torch.no_grad():
          test_loss = self.model.forward_kld(x_test)
          test_loss += self.l2_regularisation_strength * l2_regularisation
        self.test_loss_hist = np.append(self.test_loss_hist, test_loss.to('cpu').data.numpy())

        # Plot loss
        plot_loss(self.train_loss_hist,self.test_loss_hist,xlabel="Epoch")

        # Check if test loss has improved
        if test_loss + self.min_delta < self.best_test_loss:
            self.best_test_loss = test_loss
            self.epochs_without_improvement = 0
            self.best_model_state = self.model.state_dict()
        else:
            self.epochs_without_improvement += 1

        # Check for early stopping
        if self.epochs_without_improvement >= self.patience:
          print("Early stopping. No improvement in test loss for", self.patience, "epochs.")
          self.early_stopping = True

  def Train(self):

    self.data_loader.PreProcess()
    self.ndim = self.data_loader.num_columns 
    self.dataset_size = self.data_loader.num_rows_train
    self.data_loader.batch_size = self.batch_size 

    self.UpdateParameters()

    for epochs in range(self.epochs):
      print(">> Epoch",epochs)
      if not self.early_stopping:
        for it in tqdm(range(int(np.floor(self.dataset_size / self.batch_size)))):
          x = self.data_loader.Sample(tt="train",it=it)
          x_test = self.data_loader.Sample(tt="test",it=it)
          new_epoch = (it == 0)
          self.TrainIteration(x,x_test,new_epoch=new_epoch)
      else:
        break
      print("- Train Loss: "+str(round(self.train_loss_hist[-1],3))+", Test Loss: "+str(round(self.test_loss_hist[-1],3)))

    self.model.load_state_dict(self.best_model_state)

  def GenerateData(self,n_samps=1000):

    samp, log_prob = self.model.sample(n_samps)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0
    samp = self.data_loader.InverseRobustScaling(samp)
    return samp
