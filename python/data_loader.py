import pandas as pd
import torch
import gc
import numpy as np
from sklearn.model_selection import train_test_split

# This is a really simple example where I load all information into memory, can do this much better in future
class DataLoader:

  def __init__(self):

    self.full = pd.DataFrame()
    self.train = pd.DataFrame()
    self.test = pd.DataFrame()
    self.train_test_split = 0.5
    self.batch_size = 512
    self.colums = []   

    self.robust_scaling_median = None
    self.robust_scaling_iqr = None 

    self.num_rows_train = 0
    self.num_rows_test = 0
    self.num_columns = 0
    self.random_seed = 42
    torch.manual_seed(self.random_seed)


  def PreProcess(self):

    torch.manual_seed(self.random_seed)
    self.columns = list(self.full.columns)
    self.num_columns = len(self.full.columns)
    self.RobustScaling()
    self.train, self.test = train_test_split(self.full, test_size=self.train_test_split)
    del self.full
    gc.collect()
    # put in tensor
    self.train = torch.tensor(self.train.values,dtype=torch.float32)
    self.test = torch.tensor(self.test.values,dtype=torch.float32)

    random_indices = torch.randperm(len(self.train))
    self.train = self.train[random_indices]

    random_indices = torch.randperm(len(self.test))
    self.test = self.test[random_indices]

    self.num_rows_train = self.train.size(0)
    self.num_rows_test = self.test.size(0)

  def Sample(self,tt="train",it=0):

    if tt == "train":
      x = self.train[it*self.batch_size:(it+1)*self.batch_size]
    elif tt == "test":
      x = self.test[it*self.batch_size:(it+1)*self.batch_size]
    return x

  def RobustScaling(self):

    self.median = self.full.median()
    q1 = self.full.quantile(0.25)
    q3 = self.full.quantile(0.75)
    self.iqr = q3 - q1

    self.full = (self.full - self.median) / self.iqr

  def InverseRobustScaling(self,samp):

    df_out = pd.DataFrame(samp.detach().numpy(), columns=self.columns)
    df_out = df_out * self.iqr + self.median
    df_out[~df_out.isin([np.inf,-np.inf, np.nan]).any(axis=1)]   
    return df_out
