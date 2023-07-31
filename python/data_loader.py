import numpy as np
import pandas as pd
import torch
import gc
import numpy as np
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
from itertools import islice
import copy

class DataLoader:

  def __init__(self,loc,columns=None):

    self.loc = loc
    self.pf = pq.ParquetFile(loc)
    self.num_rows = self.pf.metadata.num_rows

    self.train_test_split = 0.5
    self.batch_size = 512
    self.random_seed = 42
    self.generator = None
    self.columns = columns

    samp,_ = self.LoadData(new_epoch=True)
    self.num_columns = samp.shape[1]

  def PreProcess(self,columns=None,shuffle=False):
    return None

  def LoadData(self,new_epoch=False):

    if new_epoch:
      self.generator = self.pf.iter_batches(batch_size=self.batch_size)
   
    batch = next(self.generator).to_pandas()
    if self.columns != None:
      batch = batch.loc[:,self.columns]
    x_train, x_test = train_test_split(batch, test_size=self.train_test_split, random_state=self.random_seed)
    x_train = torch.tensor(x_train.values.astype(np.float32))
    x_test = torch.tensor(x_test.values.astype(np.float32))
    return x_train, x_test

  def MakeHistograms(self,num_bins=20,ignore_quantile=0.01):

    # initial loop through to find min and max value for each histogram
  
    if self.columns != None:
      columns = self.columns
    else:
      columns = [field.name for field in self.pf.schema]
  
    min_max_quant = {k:[0,0] for k in columns}
    self.generator = self.pf.iter_batches(batch_size=self.batch_size)
    for i in range(0,int(np.floor(self.num_rows/self.batch_size))):
      batch = next(self.generator).to_pandas()
      for k in columns:
        min_max_quant[k] = [min_max_quant[k][0]+batch.loc[:,k].quantile(ignore_quantile),min_max_quant[k][1]+batch.loc[:,k].quantile(1-ignore_quantile)]
      
    for k, v in min_max_quant.items():
      min_max_quant[k] = v/np.floor(self.num_rows/self.batch_size)
    
    # second loop through to stack histograms

    bin_edges = {k:np.linspace(v[0],v[1], num_bins + 1) for k, v in min_max_quant.items()}
    hists = {"train":{},"test":{},"combined":{}}
    self.generator = self.pf.iter_batches(batch_size=self.batch_size)
    for i in range(0,int(np.floor(self.num_rows/self.batch_size))):
      batch = {}
      batch["combined"] = next(self.generator).to_pandas()
      batch["train"], batch["test"] = train_test_split(batch["combined"], test_size=self.train_test_split, random_state=self.random_seed)
      for t in hists.keys():
        for k in columns:
          if k in hists[t].keys():
            hists[t][k]+= np.histogram(batch[t].loc[:,k],bins=bin_edges[k])[0]
          else:
            hists[t][k],_ = np.histogram(batch[t].loc[:,k],bins=bin_edges[k])
      
    return bin_edges,hists