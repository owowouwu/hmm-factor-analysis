import numpy as np
import scipy as sp
import scipy.stats
from scipy.special import digamma
from hmcfa.model import HiddenMarkovFA
import logging
logging.basicConfig(level=logging.INFO)

T = 5
G = 100
N = 2000
K = 3

data = np.random.random(size = (T,G,N))
fa = HiddenMarkovFA(data = data, n_factors = K, hyperparameters={})


fa.run(max_it=5, progress_bar=True)

