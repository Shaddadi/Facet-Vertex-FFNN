import sys
sys.path.insert(0, '../../../class')

import os
import psutil
import time
import nnet
import pickle
import FFNN as ffnn
import cubelattice as cl
import multiprocessing
from functools import partial
from scipy.io import loadmat
import numpy as np
import types

def nn_reachable_sets(filemat, p, ii ,jj, lb, ub, verify):
    W = filemat['W'][0]
    b = filemat['b'][0]
    range_for_scaling = filemat['range_for_scaling'][0]
    means_for_scaling = filemat['means_for_scaling'][0]

    for i in range(5):
        lb[i] = (lb[i] - means_for_scaling[i]) / range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i]) / range_for_scaling[i]

    nnet0 = nnet.nnetwork(W, b)
    nnet0.verification = verify
    initial_input = cl.cubelattice(lb, ub).to_lattice()
    cores = multiprocessing.cpu_count()

    start_time = time.time()
    outputSets = ffnn.nnet_output(nnet0, initial_input, ("parallel", cores))

    elapsed_time = time.time() - start_time
    filename = "logs/output_info_"+str(p)+"_"+str(ii)+"_"+str(jj)+".txt"
    file = open(filename, 'w')
    file.write('time elapsed: %f seconds \n' % elapsed_time)
    file.write('number of polytopes: %d \n' % len(outputSets))
    # file.write('verification result: '+ resl+'\n')
    file.close()


if __name__ == "__main__":
    # p = int(sys.argv[1:][0])
    # i = int(sys.argv[1:][1])
    # j = int(sys.argv[1:][2])
    p = 3
    i = 1
    j = 7
    print("Property " + str(p) + "; " + "Network: N" + str(i) + "_" + str(j))

    nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    filemat = loadmat(nn_path)

    if p == 1:
        lb = [55947.691, -3.141592, -3.141592, 1145, 0]
        ub = [60760, 3.141592, 3.141592, 1200, 60]

        def verification(outputs):
            safe = True
            for afv in outputs:
                vs = afv.vertices
                if np.any(vs[0, :] >= 3.9911):
                    safe = False
                    break
            return safe

    elif p == 2:
        lb = [55947.691, -3.141592, -3.141592, 1145, 0]
        ub = [60760, 3.141592, 3.141592, 1200, 60]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmax(afv.vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == 3:
        lb = [1500, -0.06, 3.1, 980, 960]
        ub = [1800, 0.06, 3.141592, 1200, 1200]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == 4:
        lb = [1500, -0.06, 0, 1000, 700]
        ub = [1800, 0.06, 0, 1200, 800]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == 5:
        lb = [250, 0.2, -3.141592, 100, 0]
        ub = [400, 0.4, -3.141592 + 0.005, 400, 400]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx != 4):
                    safe = False
                    break
            return safe

    elif p == 6.1:
        lb = [12000, 0.7, -3.141592, 100, 0]
        ub = [62000, 3.141592, -3.141592 + 0.005, 1200, 1200]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx != 0):
                    safe = False
                    break
            return safe

    elif p == 6.2:
        lb = [12000, -3.141592, -3.141592, 100, 0]
        ub = [62000, -0.7, -3.141592 + 0.005, 1200, 1200]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx != 0):
                    safe = False
                    break
            return safe

    elif p == 7:
        lb = [0, -3.141592, -3.141592, 100, 0]
        ub = [60760, 3.141592, 3.141592, 1200, 1200]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx == 3) or np.any(indx == 4):
                    safe = False
                    break
            return safe

    elif p == 8:
        lb = [0, -3.141592, -0.1, 600, 600]
        ub = [60760, -0.75 * 3.141592, 0.1, 1200, 1200]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if (2 in indx) or (3 in indx) or (4 in indx):
                    safe = False
                    break
            return safe

    elif p == 9:
        lb = [2000, -0.4, -3.141592, 100, 0]
        ub = [7000, -0.14, -3.141592 + 0.01, 150, 150]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx != 3):
                    safe = False
                    break
            return safe

    elif p == 10:
        lb = [36000, 0.7, -3.141592, 900, 600]
        ub = [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]

        def verification(outputs):
            safe = True
            for afv in outputs:
                indx = np.argmin(afv.vertices, axis=1)
                if np.any(indx != 0):
                    safe = False
                    break
            return safe

    else:
        raise RuntimeError(f"property {p} is not defined!")

    nn_reachable_sets(filemat, p, i, j, lb, ub, verify=verification)

