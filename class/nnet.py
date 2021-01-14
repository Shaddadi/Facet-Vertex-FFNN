import numpy as np
import time
import os
import multiprocessing


class nnetwork:
    """
        A class used to represent the process of VFLs in neural networks
        ...
        Attributes
        ----------
        W : array
            weights of each layer in the network
        b : array
            bias of each layer in the network
        numLayer : int
            the number of layers
        start_time : float
            specify the starting time of computation
        filename: str
            specify the file path to save results
        compute_unsafety: bool
            determine if the unsafe inputs are computed
        unsafe_domain: array
            specify the unsafe output domain of the network

        Methods
        -------
        verification()
            a function to verify the reachable output domain

        layerOutput(inputPoly, m, lock=None):
            compute output reachable sets given a input set in m-th layer

        backtrack(vfl_sets)
            compute corresponding unsafe input sets given output sets

        singleLayerOutput(inputPoly, layerID)
            compute output reachable sets of one layer given input sets to this layer

        splitPoly(inputPoly, idx):
            split one set w.r.t. one neuron

        relu_layer(im_fl, neurons, flag=True)
            a recursive function to compute reachable sets in one layer

        get_valid_neurons(afl, neurons)
            a recursive function to identify neurons that split input sets
        """

    def __init__(self, W, b):
        """
        Parameters
        ----------
        W : array
            weights of each layer in the network
        b : array
            bias of each layer in the network
        """
        self.W = W
        self.b = b
        self.numLayer = len(W)
        self.start_time = 0
        self.filename = ''
        self.compute_unsafety = False
        self.unsafe_domain = None


    def verification(self):
        """ External verification funtions will be assigned to verify output reachable sets
        """
        print('Designed to be replaced')


    # nn output of input starting from mth layer
    def layerOutput(self, inputPoly, m, lock=None):
        """ Compute output reachable sets given a input set in m-th layer
        
        Parameters
        ----------
        inputPoly: vfl
            an input vfl
        m: int
            it indicates m-th layer
        lock: lock
            it limits multiple access to one file

        Return
        ------
        when "compute_unsafety" is set True, it returns unsafe input sets computed by "backtrack"
        when "compute_unsafety" is set True, it returns none
        """

        inputSets = [inputPoly]
        for i in range(m, self.numLayer):
            inputSets_len = len(inputSets)
            n = 0
            while n < inputSets_len:
                aPoly = inputSets.pop(0)
                n += 1
                inputSets.extend(self.singleLayerOutput(aPoly, i))

        if self.compute_unsafety:
            unsafe_inputs = self.backtrack(inputSets)
            return unsafe_inputs
        else:
            verify_result = self.verification(inputSets)
            if not verify_result:
                lock.acquire()
                try:
                    print('time elapsed: %f seconds ' % (time.time() - self.start_time))
                    print('result: unsafe\n')
                    file = open(self.filename, 'w')
                    file.write('time elapsed: %f seconds \n' % (time.time()-self.start_time))
                    file.write('result: unsafe')
                    file.close()
                finally:
                    lock.release()
                    os.system('pkill -9 python')


    def backtrack(self, vfl_sets):
        """ Compute corresponding unsafe input sets given output sets

        Parameters
        ----------
        vfl_sets: list
            output reachable sets

        Return
        ------
        vfls_unsafe: list
            unsafe input sets
        """
        matrix_A = self.unsafe_domain[0]
        vector_d = self.unsafe_domain[1]

        vfls_unsafe = vfl_sets
        for n in range(len(matrix_A)):
            A = matrix_A[[n]]
            d = vector_d[[n]]
            temp = []
            for vfl in vfls_unsafe:
                subvfl0 = vfl.single_split(A, d)
                if subvfl0:
                    temp.append(subvfl0)

            vfls_unsafe = temp

        vfls_unsafe = [vfl.vertices for vfl in vfls_unsafe]
        return vfls_unsafe


    # polytope output of single layer
    def singleLayerOutput(self, inputPoly, layerID):
        """ Compute output reachable sets of one layer given input sets to this layer

        Parameters
        ----------
        inputPoly: vfl
            an input vfl to the layer
        layerID: int
            index of the layer

        Return
        ------
        polys: list
            output reachable sets of the layer
        """
        W = self.W[layerID]
        b = self.b[layerID]
        inputPoly.linearTrans(W, b)

        # partition graph sets according to properties of the relu function
        if layerID == self.numLayer-1:
            return [inputPoly]

        polys = [inputPoly]

        splited_polys = []
        for aPoly in polys:
            splited_polys.extend(self.relu_layer(aPoly, np.array([]), flag=False))
        polys = splited_polys

        return polys


    def splitPoly(self, inputPoly, idx):
        """ Split one set w.r.t. one neuron

        Parameters
        ----------
        inputPoly: vfl
            an input vfl to the layer
        idx: int
            index of the neuron

        Return
        ------
        outputPolySets: list
            splited subsets
        """

        outputPolySets = []

        sub0, sub1= inputPoly.single_split_relu(idx)

        if sub0:
            outputPolySets.append(sub0)
        if sub1:
            outputPolySets.append(sub1)

        return outputPolySets


    def relu_layer(self, im_fl, neurons, flag=True):
        """ A recursive function to compute reachable sets in one layer

        Parameters
        ----------
        im_fl: vfl
        neurons: array
            indices of neurons

        Return
        ------
        all_fls: list
            output reachable sets of one layer
        """

        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons(im_fl, neurons)

        im_fl.map_negative_poly(new_neurons_neg)

        if new_neurons.shape[0] == 0:
            return [im_fl]

        fls = self.splitPoly(im_fl, new_neurons[0])

        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer(afl, new_neurons))

        return all_fls


    def get_valid_neurons(self, afl, neurons):
        """ A recursive function to identify neurons that split input sets

        Parameters
        ----------
        afl: vfl
        neurons: array
            indices of candidiate neurons

        Return
        ------
        valid_neurons_neg_pos: array
            neurons that split the afl
        valid_neurons_neg: array
            neurons where the afl locates in their negative domain
        """
        if neurons.shape[0] ==0:
            vertices = np.dot(afl.vertices, afl.M.T) + afl.b.T
            flag_neg = vertices<=0
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
            temp_pos = np.all(vertices>=0, 0)
            neurons_sum = temp_neg+temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T[:,0]
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = np.dot(afl.vertices,afl.M[neurons,:].T)+afl.b[neurons,:].T
        flag_neg = (elements <= 0)
        temp_neg = np.all(flag_neg, 0)
        temp_pos = np.all(elements>=0, 0)
        temp_sum = temp_neg + temp_pos
        indx_neg_pos = np.asarray(np.nonzero(temp_sum == False)).T[:,0]
        valid_neurons_neg_pos = neurons[indx_neg_pos]
        indx_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
        valid_neurons_neg = neurons[indx_neg]

        return valid_neurons_neg_pos, valid_neurons_neg

    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
