import sys
import copy as cp
import numpy as np

class VFL:
    """
    A class used to represent the data structure of Vertex-Facet Lattice

    Attributes
    ----------
    lattice : array
        a matrix to represent the lattice
    vertices : array
        vertices of a polytope
    dim : int
        the dimension of a polytope
    M : array
        a matrix for affine transformation
    b: array
        a vector for affine transformation

    Methods
    -------
    linearTrans(M, b)
        apply affine transformation on VFL

    map_negative_poly(n)
        map polytopes that locate in the negative domain of the ReLU functions

    single_split_relu(idx)
        split operation from one ReLU neuron

    single_split(A, d)
        split operation from one hyperplane
    """
    
    def __init__(self, lattice, vertices, dim, M, b):
        """
        Parameters
        ----------
        lattice : array
            a matrix to represent the lattice
        vertices : array
            vertices of a polytope
        dim : int
            the dimension of a polytope
        M : array
            a matrix for affine transformation
        b: array
            a vector for affine transformation
        """
        self.lattice = lattice
        self.vertices = vertices
        self.dim = dim
        self.M = M
        self.b = b


    def linearTrans(self, M, b):
        """ Affine transformation on VFL

        Parameters
        ----------
        M : array
            a matrix for affine transformation
        b: array
            a vector for affine transformation
        """
        if M.shape[1] != self.M.shape[0]:
            print("dimension is inconsistant")
            sys.exit(1)
        self.M = np.dot(M, self.M)
        self.b = np.dot(M, self.b) + b

        # update dim
        if M.shape[0] < self.dim:
            self.dim = M.shape[0]

        # set some dim to zero for Relu function

    def map_negative_poly(self, n):
        """ Map VFL that locate in the negative domain of the ReLU functions

        Parameters
        ----------
        n : array or int
            indices of the target ReLU neurons

        """
        if self.dim == 0:
            return self

        self.M[n, :] = 0
        self.b[n, :] = 0
        return self


    def single_split_relu(self, idx):
        """ Split operation of one ReLU neuron

        Parameters
        ----------
        idx : int
            index of the target ReLU neuron

        """
        elements = np.dot(self.vertices, self.M[idx,:].T)+self.b[idx,:].T
        if np.any(elements==0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        negative_bool = np.invert(positive_bool)
        negative_id = np.asarray(negative_bool.nonzero()).T

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
        elements0 = elements[less_bool]
        elements1 = elements[more_bool]

        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))
        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:,np.any(vs_facets0,0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = VFL(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == 1:
            subset0.map_negative_poly(idx)

        new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        new_vertices1 = np.concatenate((vertices1, new_vs))
        subset1 = VFL(sub_vs_facets1, new_vertices1, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == -1:
            subset1.map_negative_poly(idx)
        
        return subset0, subset1


    def single_split(self, A, d):
        """ Split operation of VFL from a hyperplane AX + d = 0

        Parameters
        ----------
        A : array
        d : float

        Reutrn
        ------
        A VFL locating in AX + d <= 0
        """
        elements = np.dot(np.dot(A,self.M), self.vertices.T) + np.dot(A, self.b) +d
        elements = elements[0]
        if np.all(elements >= 0):
            return None
        if np.all(elements <= 0):
            return self
        if np.any(elements == 0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements > 0)
        negative_bool = np.invert(positive_bool)


        vs_facets0 = self.lattice[negative_bool]
        vs_facets1 = self.lattice[positive_bool]
        vertices0 = self.vertices[negative_bool]
        vertices1 = self.vertices[positive_bool]
        elements0 = elements[negative_bool]
        elements1 = elements[positive_bool]

        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))
        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:, np.any(vs_facets0, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):, 0] = True  # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = VFL(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))


        return subset0




