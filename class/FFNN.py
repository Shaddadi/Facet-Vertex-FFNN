import os
import sys
import random
import numpy as np
import multiprocessing
from functools import partial

def nnet_output(nnet, inputPoly, method=("non_parallel",), output_type=0):
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

        cpus = method[1]
        outputSets = []

        nputSets0 = nnet.singleLayerOutput(inputPoly, 0)
        nputSets = []

        for apoly in nputSets0:
            nputSets.extend(nnet.singleLayerOutput(apoly, 1))

        nnet.pool = multiprocessing.Pool(cpus)
        nnet.output_type = output_type
        outputSets.extend(nnet.pool.imap(partial(nnet.layerOutput, m=2), nputSets))
        outputSets = [item for sublist in outputSets for item in sublist]
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

