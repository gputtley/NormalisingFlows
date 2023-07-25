import pandas as pd
import torch
import gc
from sklearn.model_selection import train_test_split

# This is a really simple example where I load all information into memory, can do this much better in future
class DataLoader:

  def __init__(self):

    self.full = pd.DataFrame()
    self.train = pd.DataFrame()
    self.test = pd.DataFrame()
    self.train_test_split = 0.5
    self.batch_size = 512
   
    self.robust_scaling_median = None
    self.robust_scaling_iqr = None 

    self.num_rows_train = 0
    self.num_rows_test = 0
    self.num_columns = 0
    self.random_seed = 42
    torch.manual_seed(self.random_seed)


  def PreProcess(self):

    torch.manual_seed(self.random_seed)
    self.num_columns = len(self.full.columns)
    self.RobustScaling()
    self.train, self.test = train_test_split(self.full, test_size=self.train_test_split)
    del self.full
    gc.collect()
    # put in tensor
    self.train = torch.tensor(self.train.values,dtype=torch.float32)
    self.test = torch.tensor(self.test.values,dtype=torch.float32)

    self.num_rows_train = self.train.size(0)
    self.num_rows_test = self.test.size(0)

  def Sample(self,tt="train"):

    if tt == "train":
      random_indices = torch.randperm(self.num_rows_train)[:self.batch_size]
      x = self.train[random_indices]
    elif tt == "test":
      random_indices = torch.randperm(self.num_rows_test)[:self.batch_size]
      x = self.test[random_indices]
    return x

  def RobustScaling(self):

    self.median = self.full.median()
    q1 = self.full.quantile(0.25)
    q3 = self.full.quantile(0.75)
    self.iqr = q3 - q1

    self.full = (self.full - self.median) / self.iqr


