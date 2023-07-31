import itertools
import os
import json
import argparse
import numpy as np
from batch import CreateJob, CreateBatchJob, SubmitBatchJob

parser = argparse.ArgumentParser()
parser.add_argument('--submit', help= 'Submit jobs',  action='store_true')
parser.add_argument('--collect', help= 'Collect jobs',  action='store_true')
args = parser.parse_args()

if not os.path.exists("hyperparameters"):
  os.makedirs("hyperparameters")

hyperparameters = {
    "patience" : [5],
    "min_delta" : [0.001],
    "l2_regularisation_strength" : [0.01],
    "max_grad_norm" : [100.0],
    "batch_size" : [2**(8),2**(9),2**(10)],
    "epochs" : [100],
    "learning_rate" : [5e-3,1e-3],
    "step_size" : [6],
    "gamma" : [0.1],
    "hidden_layers" : [[32],[64],[128],[256],[32,32],[64,64]],
    "num_layers" : [8,16,32,64,128],
    "masking" : ["Alternate Random"],
}

keys, values = zip(*hyperparameters.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

if args.submit:

  for ind, k in enumerate(permutations_dicts):
    
    with open("hyperparameters/model_{}.json".format(ind), 'w') as json_file:
      json.dump(k, json_file)
      
    pjob = [
      "python3 test/ggHToTauTau_example.py --no-plot --verbosity=1 --dump-model-loss --hyperparameters='hyperparameters/model_{}.json' --model-name='model_{}'".format(ind,ind),
    ]
    
    CreateBatchJob("jobs/model_{}.sh".format(ind),os.getcwd(),pjob)
    SubmitBatchJob("jobs/model_{}.sh".format(ind))
    
if args.collect:
  
  best_model = 0
  lowest_loss = np.inf
  
  for ind, k in enumerate(permutations_dicts):
    if os.path.exists("output/model_{}_loss.txt".format(ind)):
      f = open("output/model_{}_loss.txt".format(ind), "r")
      l = f.readline().rstrip()
      if l == "inf": continue
      l = float(l)
      if l < lowest_loss:
        lowest_loss = 1.0*l
        best_model = 1*ind
  print("Best model:",ind)
  print("Loss:",lowest_loss)
  os.system("python3 test/ggHToTauTau_example.py --load-model --model-name='model_{}' --hyperparameters='hyperparameters/model_{}.json'".format(ind,ind))
    