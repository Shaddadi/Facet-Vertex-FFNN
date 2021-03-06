import sys
sys.path.insert(0, '../../class')

import os
import time
import nnet
import cubelattice as cl
import multiprocessing
from functools import partial
from scipy.io import loadmat
from scipy.io import savemat
from random import shuffle
import numpy as np
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Verification Settings')
    parser.add_argument('--property', type=str, default='1')
    parser.add_argument('--n1', type=int, default=2)
    parser.add_argument('--n2', type=int, default=3)
    parser.add_argument('--compute_unsafety', action='store_true')
    args = parser.parse_args()
    p = args.property
    i = args.n1
    j = args.n2
    compute_unsafety = args.compute_unsafety

    print("Property " + p + "; " + "Network: N" + str(i) + "_" + str(j))
    if not os.path.isdir('logs'):
        os.mkdir('logs')


    if p == '1':
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

    elif p == '2':
        lb = [55947.691, -3.141592, -3.141592, 1145, 0]
        ub = [60760, 3.141592, 3.141592, 1200, 60]
        unsafe_mat = np.array([[-1.0, 1.0, 0, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 0, 1, 0], [-1, 0, 0, 0, 1]])
        unsafe_vec = np.array([[0.0], [0.0], [0.0], [0.0]])

        def verification(outputs):
            safe = True
            for afv in outputs:
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmax(vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == '3':
        lb = [1500, -0.06, 3.1, 980, 960]
        ub = [1800, 0.06, 3.141592, 1200, 1200]
        unsafe_mat = np.array([[1.0, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
        unsafe_vec = np.array([[0.0], [0], [0], [0]])

        def verification(outputs):
            safe = True
            for afv in outputs:
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == '4':
        lb = [1500, -0.06, 0, 1000, 700]
        ub = [1800, 0.06, 0.000001, 1200, 800]
        unsafe_mat = np.array([[1.0, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
        unsafe_vec = np.array([[0.0], [0], [0], [0]])

        def verification(outputs):
            safe = True
            for afv in outputs:
                vertices = np.dot(afv.vertices, afv.M.T) + afv.b.T
                indx = np.argmin(vertices, axis=1)
                if np.any(indx == 0):
                    safe = False
                    break
            return safe

    elif p == '5':
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

    elif p == '6.1':
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

    elif p == '6.2':
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

    elif p == '7':
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

    elif p == '8':
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

    elif p == '9':
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

    elif p == '10':
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
    ranges = filemat['range_for_scaling'][0]
    means = filemat['means_for_scaling'][0]

    for n in range(5):
        lb[n] = (lb[n] - means[n]) / ranges[n]
        ub[n] = (ub[n] - means[n]) / ranges[n]

    nnet0 = nnet.nnetwork(W, b)
    nnet0.verification = verification
    nnet0.compute_unsafety = compute_unsafety
    nnet0.start_time = time.time()
    nnet0.filename = "logs/output_info_" + p + "_" + str(i) + "_" + str(j) + ".txt"
    if os.path.isfile(nnet0.filename):
        os.remove(nnet0.filename)

    initial_input = cl.cubelattice(lb, ub).to_lattice()
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)
    m = multiprocessing.Manager()
    lock = m.Lock()

    unsafe_inputs = []
    nputSets0 = nnet0.singleLayerOutput(initial_input, 0)
    if p=='7' and i==1 and j==9:
        shuffle(nputSets0)
        pool.map(partial(nnet0.layerOutput, m=1, lock=lock), nputSets0)
    else:
        nputSets = []
        for apoly in nputSets0:
            nputSets.extend(nnet0.singleLayerOutput(apoly, 1))
        shuffle(nputSets)
        if compute_unsafety:
            nnet0.unsafe_domain = [unsafe_mat, unsafe_vec]
            unsafe_inputs.extend(pool.imap(partial(nnet0.layerOutput, m=2), nputSets))
            unsafe_inputs = [item for sublist in unsafe_inputs for item in sublist]
            savemat("logs/unsafe_inputs_p"+p+"_N"+str(i)+str(j)+".mat", {'unsafe_inputs':np.array(unsafe_inputs, dtype='object'), 'rangex':ranges,'meanx': means})
        else:
            pool.map(partial(nnet0.layerOutput, m=2, lock=lock), nputSets)

    pool.close()

    elapsed_time = time.time() - nnet0.start_time
    if not compute_unsafety:
        print('time elapsed: %f seconds' % elapsed_time)
        print('result: safe\n')
        file = open(nnet0.filename, 'w')
        file.write('time elapsed: %f seconds \n' % elapsed_time)
        file.write('result: safe\n')
        file.close()

