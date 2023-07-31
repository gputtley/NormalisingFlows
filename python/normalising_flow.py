import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import normflows as nf
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
from plotting import plot_loss
from data_loader import DataLoader

class NormalisingFlow:

  def __init__(self,loc,columns=None):

    ### Type of flow
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
    self.masking = "Alternate Random" # Can be "Alternate Random", "Alternate Fixed", "Random"
    self.masking_nrand = 4

    ### Dataset
    self.data_loader = DataLoader(loc,columns=columns)
   
    ### Variables defined by UpdateParameters
    self.base = None # possible I want to change the way I handle the base in future
    self.model = None
    self.best_model_state = None
    self.optimiser = None
    self.scheduler = None
    self.ndim = None
    self.dataset_size = None
    self.max_iter = None
    self.max_iter_per_epoch = None
   
    ### Variables used whilst running
    self.best_test_loss = float('inf')
    self.epochs_without_improvement = 0
    self.train_loss_hist = np.array([])
    self.test_loss_hist = np.array([])
    self.early_stopping = False
    
    ### Random seed
    self.random_seed = 42
    
    ### Other
    self.columns = columns
    self.verbosity = 2
    
  def SetParameters(self,config_dict):
    
    for key, value in config_dict.items():
      setattr(self, key, value)
    
  def UpdateParameters(self):

    self.ndim = self.data_loader.num_columns
    self.dataset_size = self.data_loader.num_rows
    self.base = nf.distributions.base.DiagGaussian(self.ndim)
    self.GetModel()
    self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularisation_strength)
    self.scheduler = lr_scheduler.StepLR(self.optimiser, step_size=self.step_size, gamma=self.gamma)
    self.best_model_state = self.model.state_dict()
    self.max_iter = int(self.epochs * np.ceil(self.dataset_size / self.batch_size))
    self.max_iter_per_epoch = self.max_iter/self.epochs
    self.dataset_size = self.data_loader.num_rows
    np.random.seed(self.random_seed)
    torch.manual_seed(self.random_seed)

  def GetModel(self):

    flows = []
    flows += [nf.flows.ActNorm(self.ndim)]
    if self.flow_type == "RealNVP":
      for i in range(self.num_layers):
        s = nf.nets.MLP([self.ndim] + self.hidden_layers + [self.ndim], init_zeros=True)
        t = nf.nets.MLP([self.ndim] + self.hidden_layers + [self.ndim], init_zeros=True)
        
        if "Alternate" in self.masking:
      
          if i % 2 == 0:
            
            if "Random" in self.masking:
              b = torch.zeros(self.ndim, dtype=torch.int)
              random_indices = torch.randperm(self.ndim)[:self.ndim//2]
              b[random_indices] = 1
            elif "Fixed" in self.masking:
              b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.ndim)])

          else:
            b = 1 - b
            
        elif "Random" in self.masking:
          b = torch.zeros(self.ndim, dtype=torch.int)
          random_indices = torch.randperm(self.ndim)[:self.masking_nrand]
          b[random_indices] = 1  
          
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        flows += [nf.flows.ActNorm(self.ndim)]

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
      if self.verbosity > 1:
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
    self.data_loader.batch_size = self.batch_size 

    self.UpdateParameters()

    for epochs in range(self.epochs):
      print(">> Epoch",epochs)
      if not self.early_stopping:
        for it in tqdm(range(int(np.floor(self.dataset_size / self.batch_size)))):
          new_epoch = (it == 0)
          x,x_test = self.data_loader.LoadData(new_epoch=new_epoch)
          self.TrainIteration(x,x_test,new_epoch=new_epoch)
      else:
        break
      
      if self.verbosity > 0:
        print("- Train Loss: "+str(round(self.train_loss_hist[-1],3))+", Test Loss: "+str(round(self.test_loss_hist[-1],3)))

    self.model.load_state_dict(self.best_model_state)

  def SaveModel(self,name):
    
    torch.save(self.model.state_dict(), name)

  def GenerateData(self,n_samps=1000):

    samp, log_prob = self.model.sample(n_samps)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0
    return samp, prob
