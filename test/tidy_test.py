from plotting import plot_histogram_with_ratio, plot_loss
from normalising_flow import NormalisingFlow
import pandas as pd

#random_seed = 42
#np.random.seed(random_seed)
#torch.manual_seed(random_seed)

normflow = NormalisingFlow()
url = "https://raw.githubusercontent.com/gputtley/ML-Assessment-3/master/tt_ggH.pkl"
normflow.data_loader.full = pd.read_pickle(url)
normflow.data_loader.full = normflow.data_loader.full.loc[:,["pt_1","pt_2","m_vis","svfit_mass","pt_tt","eta_1","eta_2","dR","dphi"]]

normflow.Train()
