"""
@author: Mario Tambos
"""
from __future__ import division, print_function
from collections import deque

import pylab

import mgng
import numpy as np
import numpy.linalg as lnp


class AMGNG:

    def __init__(self, prest_size=150, pst_size=1500):
        self.prest_size = prest_size
        self.pst_size = pst_size

        self.present = mgng.MGNG(gamma=int(prest_size / 5),
                                 lmbda=int(prest_size / 10),
                                 theta=prest_size / 10)
        self.past = mgng.MGNG(gamma=int(pst_size / 5),
                              lmbda=int(pst_size / 10),
                              theta=pst_size / 10)
        self.buffer = deque(maxlen=self.prest_size)
        self.t = 0

    def _compare_models(self):
        tot = 0.
        for pr_x in self.present.model.nodes():
            pr_x_w = self.present.get_node(pr_x)['w']
            pr_x_c = self.present.get_node(pr_x)['c']
            dist = lambda x: lnp.norm(pr_x_c - x[1]['c'])
            ps_x = min(self.past.model.nodes(data=True), key=dist)
            ps_x_w = ps_x[1]['w']
            ps_x_c = ps_x[1]['c']
            # tot += lnp.norm(pr_x_w - ps_x_w)
            tot += lnp.norm(pr_x_c - ps_x_c)
        return tot / (self.prest_size)

    def time_step(self, xt):
        self.buffer.append(xt)
        self.present.time_step(xt)
        if self.t >= self.prest_size:
            pst_xt = self.buffer.popleft()
            self.past.time_step(pst_xt)
            if self.t >= self.prest_size + self.pst_size:
                return self._compare_models()
        self.t += 1
        return 0.


def main():
    # import Oger as og
    # signal = og.datasets.mackey_glass(sample_len=1500, n_samples=1, seed=50)[0][0].flatten()
    # X = np.linspace(0, 1500, 1500)
    # mu = (X[0] + X[-1]) / 2
    # sigma = np.sqrt(mu) + 1
    # signal = np.array([np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma) for x in X])
    import pandas as pd
    signal = pd.read_csv("samples.csv", index_col="elapsed_time", parse_dates=True)
    signal = signal.resample('1S')

    amgng = AMGNG()
    scores = [0.] * len(signal)
    for t, xt in enumerate(signal.values):
        scores[t] = amgng.time_step(xt)

    pylab.subplot(2, 1, 1)
    pylab.plot(range(len(signal)), signal)
    pylab.subplot(2, 1, 2)
    pylab.plot(range(len(scores)), scores)
    pylab.show()


if __name__ == '__main__':
    main()
