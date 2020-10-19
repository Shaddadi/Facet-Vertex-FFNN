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


if __name__ == "__main__":
    # p = int(sys.argv[1:][0])
    # i = int(sys.argv[1:][1])
    # j = int(sys.argv[1:][2])
    p = 3
    i = 1
    j = 1
    print("Property " + str(p) + "; " + "Network: N" + str(i) + "_" + str(j))

    if p == 1:
        lb = [55947.691, -3.141592, -3.141592, 1145, 0]
        ub = [60760, 3.141592, 3.141592, 1200, 60]

        def verification(outputs):
            safe = True
            for afv in outputs:
                vs = np.dot(afv.vertices, afv.M.T)+afv.b.T
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmax(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == 4:
        lb = [1500, -0.06, 0, 1000, 700]
        ub = [1800, 0.06, 0.000001, 1200, 800]

        def verification(outputs):
            safe = True
            for afv in outputs:
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
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
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
                if np.any(indx != 0):
                    safe = False
                    break
            return safe

    else:
        raise RuntimeError(f"property {p} is not defined!")


    nn_path = "nnet-mat-files/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.mat"
    filemat = loadmat(nn_path)


    W = filemat['W'][0]
    b = filemat['b'][0]
    range_for_scaling = filemat['range_for_scaling'][0]
    means_for_scaling = filemat['means_for_scaling'][0]

    for n in range(5):
        lb[n] = (lb[n] - means_for_scaling[n]) / range_for_scaling[n]
        ub[n] = (ub[n] - means_for_scaling[n]) / range_for_scaling[n]

    nnet0 = nnet.nnetwork(W, b)
    nnet0.verification = verification
    initial_input = cl.cubelattice(lb, ub).to_lattice()
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)
    m = multiprocessing.Manager()
    event = m.Event()

    nnet0.start_time = time.time()
    nnet0.filename = "logs/output_info_" + str(p) + "_" + str(i) + "_" + str(j) + ".txt"

    # outputSets = nnet0.layerOutput(initial_input, 0)

    outputSets = []
    nputSets0 = nnet0.singleLayerOutput(initial_input, 0)
    nputSets = []
    for apoly in nputSets0:
        nputSets.extend(nnet0.singleLayerOutput(apoly, 1))

    # pool.map(partial(nnet0.layerOutput, m=2), nputSets)
    outputSets.extend(pool.imap(partial(nnet0.layerOutput, m=2), nputSets))
    pool.close()
    outputSets = [item for sublist in outputSets for item in sublist]

    elapsed_time = time.time() - nnet0.start_time

    file = open(nnet0.filename, 'w')
    file.write('time elapsed: %f seconds \n' % elapsed_time)
    file.write('result: safe\n')
    file.write('outputs: %d\n' %len(outputSets))
    # file.write('number of polytopes: %d \n' % len(outputSets))
    # file.write('verification result: '+ resl+'\n')
    file.close()

