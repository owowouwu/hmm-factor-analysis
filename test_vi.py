import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import digamma
from hmmfa.model import HiddenMarkovFA
from hmmfa.evals import simple_time_heatmap
import logging
logging.getLogger('hmcfa').setLevel(logging.INFO)

T = 5
G = 100
N = 2000
K = 3

data = np.random.random(size = (T,G,N))
fa = HiddenMarkovFA(data = data, n_factors = K, hyperparameters={})


elbos = fa.run(max_it=100, progress_bar=True)
plt.plot(elbos)
plt.savefig('results/a.png')
plt.clf()

simple_time_heatmap(fa.eta > 0.5, xlabel = 'Factors', ylabel = 'Genes')
plt.savefig('results/b.png')