import os
import sys
import random
import numpy as np
import multiprocessing
from functools import partial
import time

def nnet_output(nnet, inputPoly, method=("non_parallel",)):
    print("running...")
    if method[0] is not "parallel":
        outputSets = nnet.layerOutput(inputPoly, 0)
        print("Total time on split: ", nnet.c)
        return outputSets
    else:
        local_cpu = os.cpu_count()
        if method[1]>local_cpu:
            print("The number of local cores is", local_cpu)
            print("The selected number of cores is too large")
            sys.exit(1)

        # t0 = time.time()
        # nputSets0 = [inputPoly]
        # for layer in range(nnet.numLayer):
        #     nputSets = []
        #
        #     for apoly in nputSets0:
        #         nputSets.extend(nnet.singleLayerOutput(apoly, layer))
        #
        #     nputSets0 = nputSets
        #
        # print('Total time: ', time.time() - t0)
        # print('nnet.time: ', nnet.time)
        # print('nnet.time0: ', nnet.time0)
        # print('nnet.time1: ', nnet.time1)
        # print('nnet.time2: ', nnet.time2)
        # return nputSets

        t0 = time.time()
        cpus = method[1]

        outputSets = []
        nputSets0 = nnet.singleLayerOutput(inputPoly, 0)
        nputSets = []

        for apoly in nputSets0:
            nputSets.extend(nnet.singleLayerOutput(apoly, 1))

        pool = multiprocessing.Pool(cpus)
        # pool.map(partial(nnet.layerOutput, m=2), nputSets)
        outputSets.extend(pool.imap(partial(nnet.layerOutput, m=2), nputSets))
        outputSets = [item for sublist in outputSets for item in sublist]
        print('Time: ', time.time()-t0)
        return outputSets



def nnet_trackback(polySets, a, d):

    for i in range(a.shape[0]):
        inputSets = []
        for aPoly in polySets:
            poly_temp =  aPoly.intersectHalfspace(a[i, :], d[i, :])
            if poly_temp:
                inputSets.append(poly_temp)
        polySets = inputSets

    return inputSets

