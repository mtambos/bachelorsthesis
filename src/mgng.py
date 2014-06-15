"""
@author: Mario Tambos
Based on:
    Andreakis, A.; Hoyningen-Huene, N. v. & Beetz, M.
    Incremental unsupervised time series analysis using merge growing neural gas
    Advances in Self-Organizing Maps, Springer, 2009, 10-18
"""

from __future__ import print_function, division

import numpy as np
import numpy.linalg as lnp
from numpy.random import random_sample

from util import Graph


"""
d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
"""
def distances(alpha, c_t, xt, ws, cs):
    return ((1 - alpha) * lnp.norm(xt - ws, axis=1) ** 2
            + alpha * lnp.norm(c_t - cs, axis=1) ** 2)


class MGNG:

    def __init__(self, dimensions=1, alpha=0.5, beta=0.75, gamma=88,
                 delta=0.5, theta=100, eta=0.9995, lmbda=600,
                 e_w=0.05, e_n=0.0006, *args, **kwargs):
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
        self.model = Graph(nodes_num=theta, dim=dimensions)
        # 2. initialize neuron set K with 2 neurons with counter e := 0 and random weight and context vectors
        weight = random_sample(self.dimensions)
        context = random_sample(self.dimensions)
        self._add_node(weight, context)
        weight = random_sample(self.dimensions)
        context = random_sample(self.dimensions)
        self._add_node(weight, context)

    """
    find winner r := arg min_{n \in K} d_n(t)
    and second winner s := arg min_{n \in K\{r}} d_n(t)
    where d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
    """
    def find_winner_neurons(self, xt):
        dists = distances(self.alpha, self.c_t, xt, self.model.weights(), self.model.contexts())
        ind = np.argpartition(dists, 2)[:2]
        ind = ind[np.argsort(dists[ind])]
        
        r = self.model.get_node_by_matrix(ind[0])
        s = self.model.get_node_by_matrix(ind[1])

        return r, s

    """
    update neuron r and its direct topological neighbors N_r:
        w_r := w_r + \epsilon_w * (x_t - w_r)
        c_r := c_r + \epsilon_w*(C_t - c_r)
        (\forall n \in N_r)
            w_n := w_n + \epsilon_n * (x_t - w_i)
            c_n := c_n + \epsilon_n*(C_t - c_i)
    """
    def _update_neighbors(self, r, xt):
        c_t = self.c_t
        e_w = self.e_w
        e_n = self.e_n
        self.model.update_weight_and_context(r['id'], xt, c_t, e_w)
        [self.model.update_weight_and_context(n, xt, c_t, e_n)
            for n in r['neighbors']]


    def _add_node(self, weight, context, error=0):
        n = self.model.add_node(self.next_n, weight, context, error)
        self.next_n += 1
        return n

    """
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
    """
    def _create_new_neuron(self):
        errors = self.model.errors()
        ind = np.argpartition(errors, 2)[:2]
        ind = ind[np.argsort(errors[ind])]
        q = self.model.get_node_by_matrix(ind[0])

        if q['neighbors']:
            f = max((self.model.get_node(nId) for nId in q['neighbors']), key=lambda x: self.model.get_error(x['id']))
            q_weight = self.model.get_weight(q['id'])
            f_weight = self.model.get_weight(f['id'])
            q_context = self.model.get_context(q['id'])
            f_context = self.model.get_context(f['id'])
            q_error = self.model.get_error(q['id'])
            f_error = self.model.get_error(f['id'])
            l = self._add_node(weight=(q_weight + f_weight) / 2,
                               context=(q_context + f_context) / 2,
                               error=self.delta * (q_error + f_error))
            self.model.remove_edge(q['id'], f['id'])
            self.model.add_edge(q['id'], l['id'], self.gamma)
            self.model.add_edge(f['id'], l['id'], self.gamma)
            self.model.update_error(q['id'], delta=(1 - self.delta))
            self.model.update_error(f['id'], delta=(1 - self.delta))

            return l

    """
    """
    def time_step(self, xt, modify_network=True):
        # 6. find winner r and second winner s
        r, s = self.find_winner_neurons(xt)
        # 7. Ct+1 := (1 - \beta)*w_r + \beta*c_r
        r_weight = self.model.get_weight(r['id'])
        r_context = self.model.get_context(r['id'])
        c_t1 = (1 - self.beta) * r_weight + self.beta * r_context
        if modify_network:
            # 8. connect r with s: E := E \cup {(r, s)}
            # 9. age(r;s) := 0
            self.model.add_edge(r['id'], s['id'], self.gamma)

            # 10. increment counter of r: e_r := e_r + 1
            self.model.update_error(r['id'], incr=1)

            # 11. update neuron r and its direct topological neighbors:
            self._update_neighbors(r, xt)

            # 12. increment the age of all edges connected with r
            #     age_{(r,n)} := age_{(r,n)} + 1 (\forall n \in N_r )
            self.model.update_node_edges_age(r['id'], 1)

            # 13. remove old connections E := E \ {(a, b)| age_(a, b) > \gamma}
            #---> it's automatically done when an edge's age counter reaches 0

            # 14. delete all nodes with no connections.
            self.model.remove_unconnected_nodes()

            # 15. create new neuron if t mod \lambda = 0 and |K| < \theta
            if self.t % self.lmbda == 0 and len(self.model.nodes()) < self.theta:
                self._create_new_neuron()


            # 16. decrease counter of all neurons by the factor \eta:
            #    e_n := \eta * e_n (\forall n \in K)
            for k in self.model.nodes():
                self.model.update_error(k['id'], eta=self.eta)

        # 7. Ct+1 := (1 - \beta)*w_r + \beta*c_r
        self.c_t = c_t1
        # 17. t := t + 1
        self.t += 1

        return r


def main(sample_len=None):
    import Oger as og
    import pylab
    from collections import deque
    if sample_len is None:
        sample_len = 150
    signal = og.datasets.mackey_glass(sample_len=sample_len * 10,
                                      n_samples=1,
                                      seed=50)[0][0].flatten()
    signal = signal + np.abs(signal.min())
    # pylab.subplot(3, 1, 1)
    # pylab.plot(range(sample_len * 10), signal)
    # 2. initialize neuron set K with 2 neurons with counter e := 0 and random weight and context vectors
    # 3. initialize connections set E \in K * K := \empty;
    # 4. initialize global temporal context C1 := 0
    mgng = MGNG()
    # 1. time variable t := 1
    # 5. read / draw input signal xt
    # 18. if more input signals available goto step 5 else terminate
    for t, xt in enumerate(signal):
        mgng.time_step(np.array([xt]))
        if t % sample_len == 0:
            print("training: %i%%" % (10 * t / sample_len))

    errors = {}
    input_sequence = list()
    mgng.c_t = np.zeros(1)
    for t, xt in enumerate(signal):
        r = mgng.time_step(np.array([xt]), modify_network=False)
        input_sequence.append(xt)
        if t >= 30:
            input_sequence.pop(0)
        if t % sample_len == 0:
            print("error: %i%%" % (10 * t / sample_len))
        
        if r['id'] not in errors:
            errors[r['id']] = [[] for _ in range(30)]
        for i in range(1, min(t+2, 31)):
            errors[r['id']][i-1].extend(input_sequence[-i:])
            

    summary = [[0, np.zeros(i+1)] for i in range(30)]
    for i in range(30):
        for v in errors.values():
            summary[i][0] += 1
            summary[i][1] += np.std(v[i], 0)
    for i in range(30):
        summary[i][1] /= summary[i][0]

    summary = [np.sum(e)/i for i,e in summary]
    print(summary)
    pylab.subplot(2, 1, 1)
    pylab.plot(range(30), summary)

    pylab.subplot(2, 1, 2)
    nodes = mgng.model.nodes()
    pylab.plot(range(len(mgng.model.nodes())),
               [mgng.model.get_weight(n['id']) for n in nodes],
               'g',
               range(len(mgng.model.nodes())),
               [mgng.model.get_context(n['id']) for n in nodes],
               'r')
    pylab.show()


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]))
