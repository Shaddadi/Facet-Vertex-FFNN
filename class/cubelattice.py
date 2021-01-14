import itertools
import numpy as np
import vertex_facet_lattice as vfl

class cubelattice:
    """
    A class used to Vertex-Facet Lattice from constraints with upper bounds and lower bounds

    Attributes
    ----------
    dim : int
        dimension of a set
    lb : array
        lower bounds
    ub : array
        upper bounds
    M : array
        a matrix for affine transformation
    b: array
        a vector for affine transformation
    bs: array
        combined upper bounds and lower bounds
    vertices: array
        vertices of a set
    lattice: array
        a matrix for FVIM


    Methods
    -------
    to_lattice()
        covert to VFL class

    compute_lattice()
        compute the matrix for FVIM

    single_split_relu(idx)
        split operation from one ReLU neuron

    compute_vertex()
        compute vertices of a set
    """
    def __init__(self, lb, ub):
        """
        Parameters
        ----------
        lb : array
            lower bounds
        ub : array
            upper bounds
        """
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub
        self.M = np.eye(len(lb))
        self.b = np.zeros((len(lb),1))
        self.bs = np.array([lb,ub]).T

        self.vertices = self.compute_vertex()
        self.compute_lattice() # compute self.lattice

    def to_lattice(self):
        """ Covert to VFL class

        Return
        ------
        a vfl
        """
        return vfl.VFL(self.lattice, self.vertices, self.dim, self.M, self.b)

    def compute_lattice(self):
        """ compute the matrix for FVIM

        Return
        ------
        a FVIM
        """
        vertex_facets = []
        for idx, vals in enumerate(self.bs):
            for val in vals:
                vs_facet = self.vertices[:,idx]==val
                vertex_facets.append(vs_facet)

        self.lattice = np.array(vertex_facets).transpose()


    def compute_vertex(self):
        """ Compute vertices of a set

        Return
        ------
        vertices 
        """
        V = []
        for i in range(len(self.ub)):
            V.append([self.lb[i], self.ub[i]])

        return np.array(list(itertools.product(*V)))

