import numpy as np
import matplotlib.pyplot as plt
import h5py
from hmmfa.model import HiddenMarkovFA
from hmmfa.evals import simple_time_heatmap
np.random.seed(123)
import logging
h5file = 'data/synthetic/no_change/arrays.h5'

with h5py.File(h5file, 'r') as f:
    ymat = np.array(f['ymat'])

fa = HiddenMarkovFA(ymat, n_factors=6)
elbos = fa.run(max_it=1000, max_tries = 5, progress_bar=False)
plt.plot(elbos)
plt.savefig('results/no_change/elbo.png')
plt.clf()

simple_time_heatmap(fa.eta > 0.5, xlabel = 'Factors', ylabel = 'Genes')
plt.savefig('results/no_change/connectivity.png')

# save transition matrix stuff
fa.save('results/no_change/variational_params.h5')