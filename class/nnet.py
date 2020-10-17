import numpy as np
import time
import os
import sys
import psutil
import pickle
import multiprocessing
from functools import partial
from multiprocessing import get_context


class nnetwork:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.c = 0
        self.numLayer = len(W)
        self.start_time = 0
        self.filename = ''

    def verification(self):
        print('Designed to be replaced')

    # nn output of input starting from mth layer
    def layerOutput(self, inputPoly, m):
        # print('Layer: ',m)

        inputSets = [inputPoly]
        for i in range(m, self.numLayer):
            inputSets_len = len(inputSets)
            n = 0
            while n < inputSets_len:
                aPoly = inputSets.pop(0)
                n += 1
                inputSets.extend(self.singleLayerOutput(aPoly, i))

        verify_result = self.verification(inputSets)
        if not verify_result:
            print('unsafe')
            file = open(self.filename, 'w')
            file.write('time elapsed: %f seconds \n' % (time.time()-self.start_time))
            file.write('result: unsafe')
            file.close()
            os.system('pkill -9 python')

        return inputSets


    # point output of nn
    def outputPoint(self, inputPoint):
        for i in range(self.numLayer):
            inputPoint = self.singleLayerPointOutput(inputPoint, i)

        return inputPoint

    # point output of single layer
    def singleLayerPointOutput(self, inputPoint, layerID):
        W = self.W[layerID]
        b = self.b[layerID]
        layerPoint = np.dot(W, inputPoint.transpose())+b
        if layerID == self.numLayer-1:
            return layerPoint.transpose()
        else:
            layerPoint[layerPoint<0] = 0
            return layerPoint.transpose()


    # polytope output of single layer
    def singleLayerOutput(self, inputPoly, layerID):
        # print("layer", layerID)
        # inputPoly = shared_inputSets[inputSets_index]
        W = self.W[layerID]
        b = self.b[layerID]
        inputPoly.vertices = (np.dot(W, inputPoly.vertices.T) + b).T

        # partition graph sets according to properties of the relu function
        if layerID == self.numLayer-1:
                inputPoly.lattice = []
                return [inputPoly]

        polys = [inputPoly]

        splited_polys = []
        for aPoly in polys:
            splited_polys.extend(self.relu_layer(aPoly, np.array([]), flag=False))
        polys = splited_polys

        return polys


    # partition one input polytope with a hyberplane
    def splitPoly(self, inputPoly, idx):
        outputPolySets = []

        sub0, sub1= inputPoly.single_split_relu(idx)

        if sub0:
            outputPolySets.append(sub0)
        if sub1:
            outputPolySets.append(sub1)

        return outputPolySets

    def relu_layer(self, im_fl, neurons, flag=True):
        if (neurons.shape[0] == 0) and flag:
            return [im_fl]

        new_neurons, new_neurons_neg = self.get_valid_neurons(im_fl, neurons)

        im_fl.vertices[:,new_neurons_neg] = 0.0

        if new_neurons.shape[0] == 0:
            return [im_fl]

        fls = self.splitPoly(im_fl, new_neurons[0])

        new_neurons = new_neurons[1:]

        all_fls = []
        for afl in fls:
            all_fls.extend(self.relu_layer(afl, new_neurons))

        return all_fls

    def get_valid_neurons(self, afl, neurons):
        if neurons.shape[0] ==0:
            flag_neg = afl.vertices<=0
            temp_neg = np.all(flag_neg, 0)
            valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:,0]
            temp_pos = np.all(afl.vertices>=0, 0)
            neurons_sum = temp_neg+temp_pos
            valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum==False)).T[:,0]
            return valid_neurons_neg_pos, valid_neurons_neg

        elements = afl.vertices[:, neurons]
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
