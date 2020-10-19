import sys
sys.path.insert(0, '../../class')

import os
import psutil
import time
import load_nnet
import pickle
import nnet
import FFNN as ffnn
import cubelattice as cl
import multiprocessing
from functools import partial
from scipy.io import loadmat
import numpy as np
import copy as cp
from load_nnet import NNet

def load_image(n):
    filename = 'my_images_netx4/image'+str(n)
    temp = open(filename,'r')
    aline = temp.read().split(',')[:-1]
    im = [float(c) for c in aline]
    im = np.array([im])/255

    labels = 'my_images_netx4/labels'
    temp = open(labels, 'r')
    val = temp.read().split(',')[n-1]
    return im, int(val)

def simulation(num,lb,ub,im,poss):
    all_rand = []
    for i in range(len(lb)):
        temp = np.random.uniform(lb[i], ub[i], num)
        all_rand.append(temp)

    all_rand = np.array(all_rand).T
    ims = np.repeat(im, num, axis=0)
    ims[:,poss] = all_rand
    return ims

def evaluate_nnet(aNNet, inputs, label):
    # Evaluate the neural network
    numLayers = aNNet.numLayers
    weights = aNNet.weights
    biases = aNNet.biases
    for layer in range(numLayers - 1):
        inputs = np.maximum(np.dot(weights[layer], inputs) + biases[layer].reshape((len(biases[layer]), 1)),
                                0)
    outputs = np.dot(weights[-1], inputs) + biases[-1].reshape((len(biases[-1]), 1))
    if np.any(np.argmax(outputs,axis=0)!=label):
        xx = 1
    return outputs


if __name__ == "__main__":

    nn_path = "nnet-mat-files/mnist-net_256x4.nnet"
    aNNet = NNet(nn_path)
    nnet0 = nnet.nnetwork(aNNet.weights, aNNet.biases)

    n = 64 #int(sys.argv[1])
    nnet0.filename = 'logs/result'+str(n)
    [im, label] = load_image(n)

    def verification(outputs):
        safe = True
        for afv in outputs:
            vertices = np.dot(afv.vertices, afv.M.T)+afv.b.T
            indx = np.argmax(vertices, axis=1)
            if np.any(indx != label):
                safe = False
                break
        return safe

    nnet0.verification = verification

    # attack_range = np.argsort(-im)[0,:20]
    # attack_range = np.arange(10)

    attack_range = np.array([391,392,393,394,419,420,421,422])
    # attack_range = np.array([391, 392, 393, 394, 395, 396])
    # attack_range = np.array([0,1,2,3,4,5,6,7,8,9])
    lb = np.zeros(len(attack_range)).tolist()
    ub = np.ones(len(attack_range)).tolist()

    # num=100000
    # ims = simulation(num, lb, ub, im, attack_range)
    # outputs = evaluate_nnet(aNNet, ims.T, label)

    initial_input = cl.cubelattice(lb, ub).to_lattice()

    eye_temp = np.eye(len(attack_range))
    M_temp = np.zeros((len(attack_range), im.shape[1]))
    M_temp[:, attack_range] = eye_temp
    b_temp = cp.copy(im)
    b_temp[:,attack_range] = 0.0
    initial_input.M = M_temp.T
    initial_input.b = b_temp.T

    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)

    nnet0.start_time = time.time()

    # outputSets = nnet0.layerOutput(initial_input, 0)

    outputSets = []
    nputSets0 = nnet0.singleLayerOutput(initial_input, 0)
    outputSets.extend(pool.imap(partial(nnet0.layerOutput, m=1), nputSets0))
    pool.close()
    outputSets = [item for sublist in outputSets for item in sublist]

    elapsed_time = time.time() - nnet0.start_time
    print('time elapsed: %f seconds \n' % elapsed_time)
    file = open(nnet0.filename, 'w')
    file.write('time elapsed: %f seconds \n' % elapsed_time)
    file.write('result: safe\n')
    file.write('outputs: %d\n' %len(outputSets))
    # file.write('number of polytopes: %d \n' % len(outputSets))
    # file.write('verification result: '+ resl+'\n')
    file.close()

