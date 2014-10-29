'''
@author: Mario Tambos
'''
import numpy as np
import prettyplotlib as ppl

# prettyplotlib imports 
import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl
X = np.linspace(0, 100, 250)
Y = np.random.normal(loc=0.5, scale=0.05, size=len(X)/10)
Y = np.tile(Y, 10)

fig, ax = plt.subplots(1)
ax.vlines(range(0, 100, 5), ymin=Y.min(), ymax=Y.max(), colors='blue', linestyles='dotted')
ax.vlines([80], ymin=Y.min(), ymax=Y.max(), colors='red', linestyles='solid')
ppl.plot(ax, X, Y, linewidth=0.75)
fig.savefig('Desktop/plot_prettyplotlib_default.png', dpi=300)
