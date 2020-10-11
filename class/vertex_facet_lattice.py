import sys
import torch
import copy
import time
import copy as cp
import numpy as np
import collections as cln

class VFL:
    def __init__(self, lattice, vertices, vertices_init, dim):
        self.lattice = lattice
        self.vertices = vertices
        self.vertices_init = vertices_init
        self.dim = dim


    def to_cuda(self):
        self.is_cuda = True
        self.vertices = self.vertices.cuda()
        self.vertices_init = self.vertices_init.cuda()


    def single_split_relu(self, idx):
        elements = self.vertices[:,idx]
        if np.any(elements==0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        negative_bool = np.invert(positive_bool)
        negative_id = np.asarray(negative_bool.nonzero()).T
        if np.all(positive_bool):
            return self, []
        if np.all(negative_bool):
            self.vertices[:,idx] = 0.0
            return [], self
        
        if len(positive_id)>=len(negative_id):
            less_bool = negative_bool
            more_bool = positive_bool
            flg = 1
        else:
            less_bool = positive_bool
            more_bool = negative_bool
            flg = -1

        vs_facets0 = self.lattice[less_bool]
        vs_facets1 = self.lattice[more_bool]
        vertices0 = self.vertices[less_bool]
        vertices1 = self.vertices[more_bool]
        vertices_init0 = self.vertices_init[less_bool]
        vertices_init1 = self.vertices_init[more_bool]
        elements0 = elements[less_bool]
        elements1 = elements[more_bool]

        time0 = 0
        for n, facets in enumerate(vs_facets0):
            temp = np.logical_and(vs_facets1, facets)
            edge_vs_bool = np.sum(temp,axis=1)==self.dim-1 # share dim-1 facets
            p0, p1s = vertices0[n], vertices1[edge_vs_bool]
            p_init0, p_init1s = vertices_init0[n], vertices_init1[edge_vs_bool]
            elem0, elem1s = elements0[n], elements1[edge_vs_bool]
            alpha = abs(elem0) / (abs(elem0) + abs(elem1s))
            p_new = p0 + ((p1s - p0).T* alpha).T
            p_init_new = p_init0 + ((p_init1s - p_init0).T* alpha).T
            # t0 = time.time()
            if n == 0:
                new_vs = p_new
                new_vs_init = p_init_new
                new_vs_facets = temp[edge_vs_bool]
            else:
                new_vs = np.concatenate((new_vs, p_new))
                new_vs_init = np.concatenate((new_vs_init, p_init_new))
                new_vs_facets = np.concatenate((new_vs_facets,temp[edge_vs_bool]))

            # print('Sub time: ', time.time()-t0)
            # time0 = time0 + time.time() - t0

        # print('Time: ', time0)

        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:,np.any(vs_facets0,0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        new_vertices_init0 = np.concatenate((vertices_init0, new_vs_init))
        subset0 = VFL(sub_vs_facets0, new_vertices0, new_vertices_init0, self.dim)
        if flg == 1:
            subset0.vertices[:, idx] = 0.0

        new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        new_vertices1 = np.concatenate((vertices1, new_vs))
        new_vertices_init1 = np.concatenate((vertices_init1, new_vs_init))
        subset1 = VFL(sub_vs_facets1, new_vertices1, new_vertices_init1, self.dim)
        if flg == -1:
            subset1.vertices[:,idx] = 0.0

        return subset0, subset1




