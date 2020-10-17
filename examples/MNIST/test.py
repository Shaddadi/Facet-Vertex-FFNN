
import numpy as np
from load_nnet import NNet
import tensorflow_datasets as tfds
import copy as cp

def load_image(n):
    filename = 'mnist_images/image'+str(n)
    temp = open(filename,'r')
    aline = temp.read().split(',')[:-1]
    im = [float(c) for c in aline]
    im = np.array([im])/255

    labels = 'mnist_images/labels'
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


def evaluate_nnet(aNNet, inputs):
    # Evaluate the neural network
    numLayers = aNNet.numLayers
    weights = aNNet.weights
    biases = aNNet.biases
    for layer in range(numLayers - 1):
        inputs = np.maximum(np.dot(weights[layer], inputs) + biases[layer].reshape((len(biases[layer]), 1)),
                                0)
    outputs = np.dot(weights[-1], inputs) + biases[-1].reshape((len(biases[-1]), 1))
    return outputs


ims, label = tfds.as_numpy(tfds.load(
    'mnist',
    split='test',
    batch_size=-1,
    as_supervised=True,
))

nn_path = "nnet-mat-files/mnist-net_256x4.nnet"
aNNet = NNet(nn_path)

ims = ims.squeeze().reshape(10000,-1)/255
outputs = evaluate_nnet(aNNet, cp.copy(ims.T))

correct_indx = (np.argmax(outputs,axis=0)==label)
correct_outputs = outputs[:,correct_indx]
correct_ims = ims[correct_indx]
correct_label = label[correct_indx]

top2 = np.argsort(correct_outputs,axis=0)[-2:]
diffs = []
for i in range(top2.shape[1]):
    t1 = top2[:,i][1]
    t2 = top2[:,i][0]
    diff = correct_outputs[t1, i]- correct_outputs[t2,i]
    diffs.append(diff)

diffs_indx = np.argsort(np.array(diffs))[50:150]
selected_correct_ims = correct_ims[diffs_indx]
selected_correct_label = correct_label[diffs_indx]
selected_correct_outputs = correct_outputs[:, diffs_indx]
for n in range(len(selected_correct_label)):
    xx = evaluate_nnet(aNNet, cp.copy(selected_correct_ims[[0]].T))
    image = selected_correct_ims[n].tolist()
    with open('my_images/image'+str(n+1), 'w') as f:
        for item in image:
            f.write("%d," % int(item*255))

    with open('my_images/labels', 'a') as f:
        f.write("%d," %selected_correct_label[n])

