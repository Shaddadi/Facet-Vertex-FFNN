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

def nn_reachable_sets(filemat, p, ii ,jj, lb, ub,  unsafe_mat, unsafe_vec):
    W = filemat['W'][0]
    b = filemat['b'][0]
    range_for_scaling = filemat['range_for_scaling'][0]
    means_for_scaling = filemat['means_for_scaling'][0]

    for i in range(5):
        lb[i] = (lb[i] - means_for_scaling[i]) / range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i]) / range_for_scaling[i]

    norm_mat = range_for_scaling[5] * np.eye(5)
    norm_vec = means_for_scaling[5] * np.ones((5, 1))

    nnet0 = nnet.nnetwork(W, b)

    initial_input = cl.cubelattice(lb, ub).to_lattice()
	
    cores = multiprocessing.cpu_count()

    start_time = time.time()
    outputSets = ffnn.nnet_output(nnet0, initial_input, ("parallel", cores))
    # resl = verify_parallel(outputSets, norm_mat, norm_vec, unsafe_mat, unsafe_vec)

    elapsed_time = time.time() - start_time
    filename = "logs/output_info_"+str(p)+"_"+str(ii)+"_"+str(jj)+".txt"
    file = open(filename, 'w')
    file.write('time elapsed: %f seconds \n' % elapsed_time)
    file.write('number of polytopes: %d \n' % len(outputSets))
    # file.write('verification result: '+ resl+'\n')
    file.close()
    outputSets =[]

def verify_parallel(polySets, norm_mat, norm_vec, unsafe_mat, unsafe_vec):
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(8)
    unsafeSets = []
    argus = partial(verify_p, norm_mat=norm_mat, norm_vec=norm_vec,unsafe_mat=unsafe_mat,unsafe_vec=unsafe_vec)
    unsafeSets.extend(pool.map(argus, polySets))
    resl = "UNSAT(Safe)"
    if np.any(np.array(unsafeSets)):
        resl = "SAT(Unsafe)"

    return resl


def verify_p(apoly, norm_mat, norm_vec, unsafe_mat, unsafe_vec):
    norm_unsafe_mat = np.dot(np.dot(unsafe_mat,norm_mat), apoly.M)
    norm_unsafe_vec = np.dot(np.dot(unsafe_mat,norm_mat),apoly.b) + np.dot(unsafe_mat, norm_vec) - unsafe_vec

    sign_vec = np.dot(norm_unsafe_mat, np.array(apoly.vertex).transpose()) + norm_unsafe_vec
    sign_vec = np.all(sign_vec<=0, axis=0)
    violation = np.any(sign_vec)
    return violation


def main_run():
    # p = int(sys.argv[1:][0])
    # i = int(sys.argv[1:][1])
    # j = int(sys.argv[1:][2])
    p = 3
    i = 1
    j = 4
    print("Property "+str(p)+"; "+"Network: N"+str(i)+"_"+str(j))

    nn_path = "nnet-mat-files/ACASXU_run2a_"+ str(i)+ "_" + str(j)+"_batch_2000.mat"
    filemat = loadmat(nn_path)

    if p == 3:
        lb = [1500, -0.06, 3.1, 980, 960]
        ub = [1800, 0.06, 3.14, 1200, 1200]
        unsafe_mat = np.array([[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
        unsafe_vec = np.array([[0], [0], [0], [0]])
    if p == 4:
        lb = [1500, -0.06, 0, 1000, 700]
        ub = [1800, 0.06, 0, 1200, 800]
        unsafe_mat = np.array([[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
        unsafe_vec = np.array([[0], [0], [0], [0]])

    nn_reachable_sets(filemat, p, i ,j, lb, ub, unsafe_mat, unsafe_vec)


if __name__ == "__main__":
    main_run()

