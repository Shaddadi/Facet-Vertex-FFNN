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
        self.pool = multiprocessing.Pool(1) # multiprocessing

    # nn output of input starting from mth layer
    def layerOutput(self, inputPoly, m):
        # print('Layer: ',m)

        inputSets = [inputPoly]
        for i in range(m, self.numLayer):
            outputPolys = []
            for aPoly in inputSets:
                outputPolys.extend(self.singleLayerOutput(aPoly, i))
            inputSets = outputPolys

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
        for i in range(len(W)):
            splited_polys = []
            for aPoly in polys:
                splited_polys.extend(self.splitPoly(aPoly, i))

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



    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
