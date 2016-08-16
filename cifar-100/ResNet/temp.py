from scipy.io import loadmat
import os
import numpy as np
train_X = loadmat('../../Datasets/cifar-100/cifar-100-matlab/train.mat')['data']
labels = loadmat('../../Datasets/cifar-100/cifar-100-matlab/train.mat')['fine_labels']
train_Y = np.zeros([labels.shape[0],1000])
for i,j in enumerate(labels):
  train_Y[i][j] = 1
np.save('train_Y.npy', train_Y)
test_X = loadmat('../../Datasets/cifar-100/cifar-100-matlab/test.mat')['data']
labels = loadmat('../../Datasets/cifar-100/cifar-100-matlab/test.mat')['fine_labels']
test_Y = np.zeros([labels.shape[0],1000])
for i,j in enumerate(labels):
  test_Y[i][j] = 1
np.save('test_Y.npy', test_Y)

print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)