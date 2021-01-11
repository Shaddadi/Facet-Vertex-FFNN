import sys
import itertools
import torch
import numpy as np
import operator as op
from functools import reduce
import reference as rf
import vertex_facet_lattice as vfl
import collections as cln
import time

class cubelattice:

    def __init__(self, lb, ub):
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub
        self.M = np.eye(len(lb))
        self.b = np.zeros((len(lb),1))
        self.bs = np.array([lb,ub]).T

        self.vertices = self.compute_vertex(self.lb, self.ub)
        self.compute_lattice() # compute self.lattice

    def to_lattice(self): 
        return vfl.VFL(self.lattice, self.vertices, self.dim, self.M, self.b)

    def compute_lattice(self):
        vertex_facets = []
        for idx, vals in enumerate(self.bs):
            for val in vals:
                vs_facet = self.vertices[:,idx]==val
                vertex_facets.append(vs_facet)

        self.lattice = np.array(vertex_facets).transpose()


    def compute_vertex(self, lb, ub):
        V = []
        for i in range(len(ub)):
            V.append([lb[i], ub[i]])

        return np.array(list(itertools.product(*V)))

