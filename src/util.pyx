#!python
#cython: embedsignature=True
#cython: boundscheck=False
from __future__ import print_function, division

from cpython cimport bool
cimport numpy as np
import numpy as np

DTYPE = np.double
DMIN = np.finfo(np.double).min
ctypedef np.double_t DTYPE_t

cdef class Graph(object):
    cdef np.ndarray _adj_matrix
    cdef np.ndarray _weights
    cdef np.ndarray _contexts
    cdef np.ndarray _errors
    cdef unsigned int _dim
    cdef unsigned int _nodes_num
    cdef np.ndarray _matrix_indices
    cdef dict _nodes
    cdef dict _matrix_to_nodeId
    cdef unsigned int _node_counter
    
    def __init__(self, unsigned int nodes_num, unsigned int dim):
        self._dim = dim
        self._nodes_num = nodes_num
        self._adj_matrix = np.zeros((nodes_num, nodes_num), DTYPE)
        self._matrix_indices = np.zeros(nodes_num, np.bool)

        self._weights = np.array([[np.nan]*dim]*nodes_num, dtype=DTYPE)
        self._contexts = np.array([[np.nan]*dim]*nodes_num, dtype=DTYPE)
        self._errors = np.array([np.nan]*nodes_num, dtype=DTYPE)

        self._nodes = dict()
        self._matrix_to_nodeId = dict()
        self._node_counter = 0
    
    cpdef list nodes(self):
        return self._nodes.values()

    cpdef np.ndarray[DTYPE_t, ndim=2] weights(self):
        return self._weights

    cpdef np.ndarray[DTYPE_t, ndim=1] get_weight(self, unsigned int id):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        return self._weights[index]

    cpdef update_weight(self, unsigned int id,
                        np.ndarray[DTYPE_t, ndim=1] xt,
                        DTYPE_t e):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        self._weights[index] += e * (xt - self._weights[index])

    cpdef np.ndarray[DTYPE_t, ndim=2] contexts(self):
        return self._contexts

    cpdef np.ndarray[DTYPE_t, ndim=1] get_context(self, unsigned int id):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        return self._contexts[index]

    cpdef update_context(self, unsigned int id,
                         np.ndarray[DTYPE_t, ndim=1] c_t,
                         DTYPE_t e):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        self._contexts[index] += e * (c_t - self._contexts[index])

    cpdef update_weight_and_context(self, unsigned int id,
                                    np.ndarray[DTYPE_t, ndim=1] xt,
                                    np.ndarray[DTYPE_t, ndim=1] c_t,
                                    DTYPE_t e_w, DTYPE_t e_n):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        self._weights[index] += e_w * (xt - self._weights[index])
        self._contexts[index] += e_w * (c_t - self._contexts[index])
        cdef list n_indices = [self._nodes[n]['matrix_index'] for n in self._nodes[id]['neighbors']]
        self._weights[n_indices] += e_w * (xt - self._weights[n_indices])
        self._contexts[n_indices] += e_w * (c_t - self._contexts[n_indices])


    cpdef np.ndarray[DTYPE_t, ndim=1] errors(self):
        return self._errors

    cpdef np.ndarray[DTYPE_t, ndim=1] get_errors(self, np.ndarray[unsigned int, ndim=1] ids):
        cdef list indices = [self._nodes[id]['matrix_index'] for id in ids]
        return self._errors[indices]

    cpdef DTYPE_t get_error(self, unsigned int id):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        return self._errors[index]

    cpdef update_error(self, unsigned int id, int incr=0,
                       DTYPE_t delta=1, DTYPE_t eta=1):
        cdef unsigned int index = self._nodes[id]['matrix_index']
        self._errors[index] = self._errors[index] * delta * eta + incr

    cpdef update_errors(self, DTYPE_t eta):
        self._errors *= eta

    cpdef dict add_node(self, unsigned int id,
                        np.ndarray[DTYPE_t, ndim=1] weight,
                        np.ndarray[DTYPE_t, ndim=1] context,
                        DTYPE_t error):
        if id in self._nodes:
            raise ValueError("node id already exists")
        if self._node_counter >= self._nodes_num:
            raise ValueError("too many nodes")

        cdef unsigned int matrix_index = self._matrix_indices.argmin()
        self._matrix_indices[matrix_index] = True
        cdef dict node = {'id': id, 'matrix_index': matrix_index,
                          'neighbors':set()}
        self._nodes[id] = node
        self._weights[matrix_index] = weight
        self._contexts[matrix_index] = context
        self._errors[matrix_index] = error
        self._matrix_to_nodeId[matrix_index] = id
        self._node_counter += 1
        return node

    cpdef dict get_node(self, unsigned int id):
        return self._nodes[id]

    cpdef dict get_node_by_matrix(self, unsigned int index):
        return self._nodes[self._matrix_to_nodeId[index]]

    cpdef remove_node(self, unsigned int id):
        cdef unsigned int matrix_index
        if id in self._nodes:
            matrix_index = self._nodes[id]['matrix_index']
            self._adj_matrix[matrix_index] = np.zeros(self._dim)
            self._adj_matrix[:, matrix_index] = np.zeros(self._dim)
            self._matrix_indices[matrix_index] = False

            self._weights[matrix_index] = np.array([np.nan] * self._dim, dtype=DTYPE)
            self._contexts[matrix_index] = np.array([np.nan] * self._dim, dtype=DTYPE)
            self._errors[matrix_index] = np.nan
            del self._matrix_to_nodeId[matrix_index]

            del self._nodes[id]
            self._node_counter += 1

    cpdef remove_unconnected_nodes(self):
        [self.remove_node(<unsigned int> k) for k in self._nodes
                                                if not self._nodes[k]['neighbors']]

    cpdef update_edge_age(self, unsigned int nodeId1, unsigned int nodeId2,
                         int age_delta, bool hard_set=False):
        cdef unsigned int matrix_index_1 = self._nodes[nodeId1]['matrix_index']
        cdef unsigned int matrix_index_2 = self._nodes[nodeId2]['matrix_index']
        if hard_set:
            self._adj_matrix[matrix_index_1, matrix_index_2] = age_delta
            self._adj_matrix[matrix_index_2, matrix_index_1] = age_delta
        else:
            self._adj_matrix[matrix_index_1, matrix_index_2] -= age_delta
            self._adj_matrix[matrix_index_2, matrix_index_1] -= age_delta

    cpdef update_node_edges_age(self, unsigned int nodeId, int age_delta):
        cdef unsigned int matrix_index_1 = self._nodes[nodeId]['matrix_index']
        [self.update_edge_age(nodeId, nId, age_delta)
            for nId in self._nodes[nodeId]['neighbors']]

    cpdef add_edge(self, unsigned int nodeId1, unsigned int nodeId2,
                   unsigned int age_limit):
        if nodeId1 == nodeId2:
            raise ValueError("cannot connect node to itself")
        self._nodes[nodeId1]['neighbors'].add(nodeId2)
        self._nodes[nodeId2]['neighbors'].add(nodeId1)
        self.update_edge_age(nodeId1, nodeId2, age_limit, True)

    cpdef remove_edge(self, unsigned int nodeId1, unsigned int nodeId2):
        self._nodes[nodeId1]['neighbors'].remove(nodeId2)
        self._nodes[nodeId2]['neighbors'].remove(nodeId1)
        self.update_edge_age(nodeId1, nodeId2, 0, True)

    cpdef list neighbors(self, unsigned int nodeId):
        return [self._nodes[n] for n in self._nodes[nodeId]['neighbors']]
