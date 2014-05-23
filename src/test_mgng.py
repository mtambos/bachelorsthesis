"""
@author: Mario Tambos
"""
from __future__ import print_function, division

import unittest

import mgng
import networkx as nx
import numpy as np


class TestMGNG(unittest.TestCase):

    def setUp(self):
        self.inst = mgng.MGNG(dimensions=1, alpha=0.13, beta=0.7, delta=0.7,
                              gamma=5, theta=100, eta=0.7, lmbda=5, e_b=1,
                              e_n=1)

    def tearDown(self):
        del self.inst

    """
    expected result: 14.44 = (1 - 0.13) * ||7 - 3||^2 + 0.13*||3 - 5||^2
    """
    def test_distance(self):
        xt = np.array(7)
        n = {'w': np.array(3), 'c': np.array(5)}
        self.inst.c_t = 3
        actual = self.inst.distance(xt, n)
        self.assertEquals(14.44, actual)

    """
    expected result: node added to model
    """
    def test_add_node(self):
        actual = self.inst._add_node(e=1, w=2, c=3)
        self.assertEquals(2, actual['i'])
        self.assertEquals(1, actual['e'])
        self.assertEquals(2, actual['w'])
        self.assertEquals(3, actual['c'])
        self.assertEquals(3, self.inst.next_n)
        self.assertEquals(2, self.inst.model.node[2]['i'])
        self.assertEquals(1, self.inst.model.node[2]['e'])
        self.assertEquals(2, self.inst.model.node[2]['w'])
        self.assertEquals(3, self.inst.model.node[2]['c'])

    """
    expected result: 1st Neuron.i = 0, 2nd Neuron.i = 1
    """
    def test_find_winners(self):
        xt = np.array(7)
        expected = [self.inst._add_node(e=0, w=np.array(3), c=np.array(5)),
                    self.inst._add_node(e=0, w=np.array(1), c=np.array(3))]
        self.inst._add_node(e=0, w=np.array(23), c=np.array(29))
        self.inst._add_node(e=0, w=np.array(17), c=np.array(19))
        self.inst.c_t = 3

        actual = self.inst.find_winner_neurons(xt)
        self.assertEquals(expected[0], actual[0])
        self.assertEquals(expected[1], actual[1])

    """
    expected result:
        r[w] = 3 + 0.7 * (7 - 3) = 5.8
        r[c] = 5 + 0.7 * (11 - 5) = 9.2
        N1[w] = 23 + 0.05 * (7 - 23) = 22.2
        N1[c] = 29 + 0.05 * (11 - 29) = 28.1
        N2[w] = 17 + 0.05 * (7 - 17) = 16.5
        N2[c] = 19 + 0.05 * (11 - 19) = 18.6
    """
    def test_update_neighbors(self):
        xt = np.array(7.)
        r = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        u = self.inst._add_node(e=0, w=np.array(23.), c=np.array(29.))
        v = self.inst._add_node(e=0, w=np.array(17.), c=np.array(19.))
        self.inst._add_edge(r, u)
        self.inst._add_edge(r, v)
        self.inst.c_t = np.array(11.)
        self.inst.e_w = 0.7
        self.inst.e_n = 0.05

        self.inst._update_neighbors(r, xt)
        self.assertEquals(5.8, r['w'])
        self.assertEquals(9.2, r['c'])
        self.assertEquals([u['i'], v['i']], self.inst.model.neighbors(r['i']))
        self.assertEquals(22.2, self.inst.model.node[u['i']]['w'])
        self.assertEquals(28.1, self.inst.model.node[u['i']]['c'])
        self.assertEquals(16.5, self.inst.model.node[v['i']]['w'])
        self.assertEquals(18.6, self.inst.model.node[v['i']]['c'])

    """
    expected resut: 1st edge.age = 3; 2nd edge.age = 4
    """
    def test_increment_edges_age(self):
        r = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        u = self.inst._add_node(e=0, w=np.array(23.), c=np.array(29.))
        v = self.inst._add_node(e=0, w=np.array(17.), c=np.array(19.))
        self.inst.model.add_edge(r['i'], u['i'], age=2)
        self.inst.model.add_edge(r['i'], v['i'], age=3)

        self.inst._increment_edges_age(r)
        self.assertEquals(3, self.inst.model[r['i']][u['i']]['age'])
        self.assertEquals(4, self.inst.model[r['i']][v['i']]['age'])

    """
    expected resut: 1 edge added between r and s, with age=0
    """
    def test_add_edge(self):
        r = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        s = self.inst._add_node(e=0, w=np.array(23.), c=np.array(29.))

        self.inst._add_edge(r, s)
        self.assertEquals((r['i'], s['i'], {'age': 0}),
                          self.inst.model.edges(data=True)[0])

    """
    expected result: edge 0 not deleted, edge 1 present
    """
    def test_remove_old_edges(self):
        r = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        s = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        t = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        u = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        self.inst._add_edge(r, s)
        self.inst._add_edge(t, u)
        self.inst.model[r['i']][s['i']]['age'] = 6

        self.inst._remove_old_edges()
        self.assertNotIn((r['i'], s['i']), self.inst.model.edges())
        self.assertIn((t['i'], u['i']), self.inst.model.edges())

    """
    """
    def test_remove_unconnected_neurons(self):
        r = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        s = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        t = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        u = self.inst._add_node(e=0, w=np.array(3.), c=np.array(5.))
        self.inst._add_edge(t, u)

        self.inst._remove_unconnected_neurons()
        self.assertNotIn(r['i'], self.inst.model.nodes())
        self.assertNotIn(s['i'], self.inst.model.nodes())
        self.assertNotIn((r['i'], s['i']), self.inst.model.edges())
        self.assertIn(t['i'], self.inst.model.nodes())
        self.assertIn(u['i'], self.inst.model.nodes())
        self.assertIn((t['i'], u['i']), self.inst.model.edges())

    """
    expected result:
        create new node if t mod \lambda = 0 and |K| < \theta
        new neuron created between neuron q with greatest e_q and
        q's neibouring neuron f with greatest e_f
        K := K \cup l
        w_l := 1/2 * (w_q + w_f)
        c_l := 1/2 * (c_q + c_f)
        e_l := \delta * (e_f + e_q)
        e_q := (1 - \deta) * e_q
        e_f := (1 - \deta) * e_f
    """
    def test_create_new_neuron(self):
        q = self.inst._add_node(e=5, w=np.array(3.), c=np.array(5.))
        f = self.inst._add_node(e=3, w=np.array(3.), c=np.array(5.))
        s = self.inst._add_node(e=2, w=np.array(3.), c=np.array(5.))
        t = self.inst._add_node(e=4, w=np.array(3.), c=np.array(5.))
        u = self.inst._add_node(e=1, w=np.array(3.), c=np.array(5.))
        self.inst._add_edge(q, s)
        self.inst._add_edge(q, f)
        self.inst._add_edge(t, u)

        expt_next_n = self.inst.next_n + 1
        expt_w_l = (q["w"] + f["w"]) / 2.
        expt_c_l = (q["c"] + f["c"]) / 2.
        expt_e_l = self.inst.delta * (f["e"] + q["e"])
        expt_e_q = (1 - self.inst.delta) * q["e"]
        expt_e_f = (1 - self.inst.delta) * f["e"]
        l = self.inst._create_new_neuron()

        self.assertEquals(expt_next_n, self.inst.next_n)
        self.assertEquals(l, self.inst.model.node[l['i']])
        self.assertEquals(2, len(self.inst.model.neighbors(l['i'])))
        self.assertIn(q['i'], self.inst.model.neighbors(l['i']))
        self.assertIn(f['i'], self.inst.model.neighbors(l['i']))
        self.assertEquals(expt_w_l, l["w"])
        self.assertEquals(expt_c_l, l["c"])
        self.assertEquals(expt_e_l, l["e"])
        self.assertEquals(expt_e_q, q["e"])
        self.assertEquals(expt_e_f, f["e"])

    """
    expected result:
        * new edge created between r and s
        * number of neurons = 3
    """
    def test_time_step(self):
        q = self.inst._add_node(e=5, w=np.array(23.), c=np.array(25.))
        r = self.inst._add_node(e=5, w=np.array(3.), c=np.array(5.))
        s = self.inst._add_node(e=3, w=np.array(5.), c=np.array(7.))
        self.inst.c_t = 3

        self.inst.t = 5
        self.inst.time_step(np.array(2.))
        self.assertEquals(3, len(self.inst.model.nodes()))
        self.assertEquals(2, len(self.inst.model.edges()))

if __name__ == '__main__':
    unittest.main()
