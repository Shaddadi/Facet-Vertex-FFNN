import numpy as np
import time
import os
import sys
import multiprocessing
from multiprocessing import get_context
from multiprocessing import Process, Queue
from random import shuffle

class nnetwork:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.c = 0
        self.numLayer = len(W)
        self.start_time = 0
        self.filename = ''
        self.test_time = 0
        self.facetv_time0 = 0
        self.facetv_time1 = 0

    def verification(self):
        print('Designed to be replaced')

    def layerOutput(self, inputPoly, m):
        # print('Layer: %d\n'%m)
        inputSets = [inputPoly]
        for i in range(m, self.numLayer):
            output = Queue()
            processes = []
            for aPoly in inputSets:
                processes.append(Process(target=self.singleLayerOutput, args=(aPoly, i, output)))

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            inputSets = [output.get() for p in processes]

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

        W = self.W[layerID]
        b = self.b[layerID]
        inputPoly.linearTrans(W, b)

        if layerID == self.numLayer - 1:
            inputPoly.lattice = []
            verify_result = self.verification(inputPoly)
            if not verify_result:
                if not os.path.isfile(self.filename):
                    print('Time elapsed: %f seconds' % (time.time()-self.start_time))
                    print('Result: unsafe \n')
                    file = open(self.filename, 'w')
                    file.write('time elapsed: %f seconds \n' % (time.time()-self.start_time))
                    file.write('result: unsafe')
                    file.close()

                    os.system('pkill -9 python')

            return

        polys = self.relu_layer(inputPoly, np.array([]), flag=False)
        shuffle(polys)
        if layerID <1:
            processes = []
            for aPoly in polys:
                processes.append(Process(target=self.singleLayerOutput, args=(aPoly, layerID+1)))

            for p in processes:
                p.start()
            for p in processes:
                p.join()
        else:
            for aPoly in polys:
                self.singleLayerOutput(aPoly, layerID + 1)

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

        t0 =time.time()
        new_neurons, new_neurons_neg = self.get_valid_neurons(im_fl, neurons)
        self.test_time = self.test_time + time.time() - t0

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
