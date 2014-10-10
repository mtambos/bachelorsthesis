'''
@author: Mario Tambos
Based on:
    Andreakis, A.; Hoyningen-Huene, N. v. & Beetz, M.
    Incremental unsupervised time series analysis using merge growing neural gas
    Advances in Self-Organizing Maps, Springer, 2009, 10-18
'''

from __future__ import print_function, division

from collections import defaultdict
from functools import partial

from numpy.random import random_sample

import networkx as nx
import numpy as np
import numpy.linalg as lnp
import numexpr as ne


class MGNG:

    def __init__(self, dimensions=1, alpha=0.5, beta=0.75, gamma=88,
                 delta=0.5, theta=100, eta=0.9995, lmbda=600,
                 e_w=0.05, e_n=0.0006):
        self.dimensions = dimensions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        self.eta = eta
        self.lmbda = lmbda
        self.e_w = e_w
        self.e_n = e_n
        # 4. initialize global temporal context C1 := 0
        self.c_t = np.zeros(dimensions)
        self.next_n = 0
        # 1. time variable t := 0
        self.t = 0
        # 3. initialize connections set E \in K * K := \empty;
        self.model = nx.Graph()
        # 2. initialize neuron set K with 2 neurons with counter e := 0
        # and random weight and context vectors
        self._add_node()
        self._add_node()

    def distance(self, xt, n):
        '''
        d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
        '''
        total = ((1 - self.alpha) * (xt - n['w'])**2 +
                 self.alpha * (self.c_t - n['c'])**2)
        return total[0]

    def find_winner_neurons(self, xt):
        '''
        find winner r := arg min_{n \in K} d_n(t)
        and second winner s := arg min_{n \in K\{r}} d_n(t)
        where d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
        '''
        dists = [(self.distance(xt, n), n) for n in self.model.node.values()]
        dists.sort()
        return dists[:2]

    def _update_neighbors(self, r, xt):
        '''
        update neuron r and its direct topological neighbors N_r:
            w_r := w_r + \epsilon_w * (x_t - w_r)
            c_r := c_r + \epsilon_w*(C_t - c_r)
            (\forall n \in N_r)
                w_n := w_n + \epsilon_n * (x_t - w_i)
                c_n := c_n + \epsilon_n*(C_t - c_i)
        '''
        r['w'] += self.e_w * (xt - r['w'])
        r['c'] += self.e_w * (self.c_t - r['c'])
        for n in self.model.neighbors(r['i']):
            n = self.model.node[n]
            n['w'] += self.e_n * (xt - n['w'])
            n['c'] += self.e_n * (self.c_t - n['c'])

    def _increment_edges_age(self, r):
        '''
        increment the age of all edges connected with r
            age_{(r,n)} := age_{(r,n)} + 1 (\forall n \in N_r )
        '''
        for (u, v) in self.model.edges(r['i']):
            self.model[u][v]['age'] += 1

    def _add_node(self, e=0, w=None, c=None):
        if w is None:
            w = random_sample(self.dimensions)
        if c is None:
            c = random_sample(self.dimensions)
        n = {'i': self.next_n, 'e': e, 'w': w, 'c': c}
        self.model.add_node(self.next_n, n)
        self.next_n += 1
        return n

    def get_node(self, i):
        return self.model.node[i]

    def _add_edge(self, r, s):
        if r == s:
            raise Exception('cannot connect edge to itself')
        if s['i'] in self.model.neighbors(r['i']):
            self.model[r['i']][s['i']]['age'] = 0
        else:
            self.model.add_edge(r['i'], s['i'], age=0)

    def get_edge(self, u, v=None):
        if v is None:
            return self.model[u]
        else:
            return self.model[u][v]

    def _remove_old_edges(self):
        '''
        remove old connections E := E \ {(a, b)| age_(a, b) > \gamma}
        '''
        for (u, v) in self.model.edges():
            if self.model.edge[u][v]['age'] > self.gamma:
                self.model.remove_edge(u, v)

    def _remove_unconnected_neurons(self):
        '''
        '''
        for n in self.model.nodes():
            if not self.model.degree(n):
                self.model.remove_node(n)

    def _create_new_neuron(self):
        '''
        create new neuron if t mod \lambda = 0 and |K| < \theta
            a. find neuron q with the greatest counter: q := arg max_{n \in K} e_n
            b. find neighbor f of q with f := arg max_{n \in N_q} e_n
            c. initialize new neuron l
                K := K \cup l
                w_l := 1/2 * (w_q + w_f)
                c_l := 1/2 * (c_q + c_f)
                e_l := \delta * (e_f + e_q)
            d. adapt connections: E := (E \ {(q, f)}) \cup {(q, n), (n, f)}
            e. decrease counter of q and f by the factor \delta
                e_q := (1 - \deta) * e_q
                e_f := (1 - \deta) * e_f
        '''
        q = max(self.model.nodes(data=True), key=lambda n: n[1]['e'])[1]
        f = max(self.model.neighbors(q['i']),
                key=lambda n: self.model.node[n]['e'])
        if f is not None:
            f = self.model.node[f]
            l = self._add_node(e=self.delta * (q['e'] + f['e']),
                               w=(q['w'] + f['w']) / 2,
                               c=(q['c'] + f['c']) / 2)
            self.model.remove_edge(q['i'], f['i'])
            self._add_edge(q, l)
            self._add_edge(f, l)
            q['e'] *= (1 - self.delta)
            f['e'] *= (1 - self.delta)

            return l

    def time_step(self, xt):
        '''
        '''
        # 6. find winner r and second winner s
        r, s = self.find_winner_neurons(xt)
        r_dist, r = r
        s_dist, s = s

        # 7. Ct+1 := (1 - \beta)*w_r + \beta*c_r
        c_t1 = (1 - self.beta) * r['w'] + self.beta * r['c']

        # 8. connect r with s: E := E \cup {(r, s)}
        # 9. age(r;s) := 0
        self._add_edge(r, s)

        # 10. increment counter of r: e_r := e_r + 1
        r['e'] += 1

        # 11. update neuron r and its direct topological neighbors:
        self._update_neighbors(r, xt)

        # 12. increment the age of all edges connected with r
        self._increment_edges_age(r)

        # 13. remove old connections E := E \ {(a, b)| age_(a, b) > \gamma}
        self._remove_old_edges()

        # 14. delete all nodes with no connections.
        self._remove_unconnected_neurons()

        # 15. create new neuron if t mod \lambda = 0 and |K| < \theta
        if self.t % self.lmbda == 0 and len(self.model.nodes()) < self.theta:
            self._create_new_neuron()

        # 16. decrease counter of all neurons by the factor \eta:
        #    e_n := \eta * e_n (\forall n \in K)
        for k in self.model.nodes():
            self.model.node[k]['e'] *= self.eta

        # 7. Ct+1 := (1 - \beta)*w_r + \beta*c_r
        self.c_t = c_t1

        # 17. t := t + 1
        self.t += 1
        
        return r_dist


def main():
    import Oger as og
    import pylab
    signal = og.datasets.mackey_glass(sample_len=1500,
                                      n_samples=1,
                                      seed=50)[0][0].flatten()
    print(signal)
    signal = signal + np.abs(signal.min())
    print(signal)
    # 2. initialize neuron set K with 2 neurons with counter e := 0 and random weight and context vectors
    # 3. initialize connections set E \in K * K := \empty;
    # 4. initialize global temporal context C1 := 0
    mgng = MGNG(lmbda=6)
    # 1. time variable t := 1
    # 5. read / draw input signal xt
    # 18. if more input signals available goto step 5 else terminate
    for t, xt in enumerate(signal, 1):
        mgng.time_step(xt)
        if t % 1500 == 0:
            print('training: %i%%' % (t / 1500))

    errors = [[] for _ in range(30)]
    for t, xt in enumerate(signal, 1):
        if t % 150 == 0:
            print('calculating errors: %i%%' % (t / 150))
        n, _ = mgng.find_winner_neurons(xt)
        n = n[1] 
        for i in range(min(30, t)):
            errors[i].append((n['w'] - signal[t - i - 1]) ** 2)

    summary = [0] * 30
    for i in range(30):
        summary[i] = np.sum(errors[i]) / len(errors[i])

    pylab.subplot(2, 1, 1)
    pylab.plot(range(30), summary)

    pylab.subplot(2, 1, 2)
    pylab.plot(range(len(mgng.model.nodes())),
               [n[1]['w'] for n in mgng.model.nodes(data=True)])
    pylab.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
