from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt
import os

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
from scipy.io import loadmat
from tensorflow.contrib.layers import layers
import numpy as np


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_boolean('use_grayscale', True,
                            """1024 Input Tensor instead of 3072, for low processing GPUs""")

def get_cifar100(x, y, batch_size, batch_i):
  return x[(batch_i*batch_size):(batch_size*(batch_i+1)), :], y[(batch_i*batch_size):(batch_size*(batch_i+1)), :]

def res_net(x, y, activation=tf.nn.relu):

  NetworkLayers = namedtuple(
      'NetworkLayers', ['repetitions', 'filters', 'layer_size'])
  clusters = [NetworkLayers(10, 32, 128),
              NetworkLayers(10, 48, 128),
              NetworkLayers(10, 64, 128),
              NetworkLayers(10, 80, 128),
              NetworkLayers(10, 96, 128),
              NetworkLayers(10, 112, 128),
              NetworkLayers(10, 128, 128),
              NetworkLayers(10, 144, 128),
              NetworkLayers(10, 160, 128),
              NetworkLayers(10, 176, 128),
              NetworkLayers(10, 192, 128),
              NetworkLayers(10, 208, 128),
              NetworkLayers(10, 224, 128),
              NetworkLayers(10, 240, 128),
              NetworkLayers(10, 256, 128)]

  input_shape = x.get_shape().as_list()

  # x[2D] -> x[4D]
  input_shape = x.get_shape().as_list()
  if len(input_shape) == 2:
    if FLAGS.use_grayscale==True:
      ndim = int(sqrt(input_shape[1]))
      if ndim * ndim != input_shape[1]:
          raise ValueError('input_shape should be square')
      x = tf.reshape(x, [-1, ndim, ndim, 1])
    else:
      ndim = int(sqrt(input_shape[1]/3))
      if ndim * ndim != input_shape[1]/3:
          raise ValueError('input_shape should be square')
      x = tf.reshape(x, [-1, ndim, ndim, 3])

  # 64 features
  with tf.variable_scope('conv_layer1'):
    net = learn.ops.conv2d(x, 64, [3, 3], batch_norm=False,
                           activation=activation, bias=False, padding='VALID')
  


  # Max pool
  net = tf.nn.max_pool(
      net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  # First chain of resnets
  with tf.variable_scope('conv_layer2'):
    net = learn.ops.conv2d(net, clusters[0].filters,
                           [1, 1], [1, 1, 1, 1],
                           padding='VALID', bias=True)

  # Create the bottleneck clusters, each of which contains `repetitions`
  # bottleneck clusters.
  for group_i, group in enumerate(clusters):
    for block_i in range(group.repetitions):
      name = 'group_%d/block_%d' % (group_i, block_i)

      # 1x1 convolution responsible for reducing dimension
      with tf.variable_scope(name + '/conv_in'):
        conv = learn.ops.conv2d(net, group.layer_size,
                                [1, 1], [1, 1, 1, 1],
                                padding='VALID',
                                activation=activation,
                                batch_norm=False,
                                bias=False)

      with tf.variable_scope(name + '/conv_bottleneck'):
        conv = learn.ops.conv2d(conv, group.layer_size,
                                [3, 3], [1, 1, 1, 1],
                                padding='SAME',
                                activation=activation,
                                batch_norm=False,
                                bias=False)

      # 1x1 convolution responsible for restoring dimension
      with tf.variable_scope(name + '/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = learn.ops.conv2d(conv, input_dim,
                                [1, 1], [1, 1, 1, 1],
                                padding='VALID',
                                activation=activation,
                                batch_norm=False,
                                bias=False)

      # shortcut connections that turn the network into its counterpart
      # residual function (identity shortcut)
      net = conv + net

    try:
      # upscale to the next group size
      next_group = clusters[group_i + 1]
      with tf.variable_scope('block_%d/conv_upscale' % group_i):
        net = learn.ops.conv2d(net, next_group.filters,
                               [1, 1], [1, 1, 1, 1],
                               bias=False,
                               padding='SAME')
    except IndexError:
      pass

  net_shape = net.get_shape().as_list()
  net = tf.nn.avg_pool(net,
                       ksize=[1, net_shape[1], net_shape[2], 1],
                       strides=[1, 1, 1, 1], padding='VALID')

  net_shape = net.get_shape().as_list()
  net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

  return learn.models.logistic_regression(net, y)

def test():

    if FLAGS.use_grayscale:
      x = tf.placeholder(tf.float32, [None, 1024])
    else:
      x = tf.placeholder(tf.float32, [None, 3072])
    y = tf.placeholder(tf.float32, [None, 100])
    y_pred, cross_entropy = res_net(x, y)

    # %% Define loss/eval/training functions
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # %% Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # %% We now create a new session to actually perform the initialization the
    # variables:
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # %% We'll train in minibatches and report accuracy:
    batch_size = 20
    n_epochs = 5

    if(os.path.isfile('train_Y.npy') and os.path.isfile('test_Y.npy')):
      train_Y = np.load('train_Y.npy')
      test_Y = np.load('test_Y.npy')
    else:
      labels = loadmat('../../Datasets/cifar-100/cifar-100-matlab/train.mat')['fine_labels']
      train_Y = np.zeros([labels.shape[0],100])
      for i,j in enumerate(labels):
       train_Y[i][j] = 1
      np.save('train_Y.npy', train_Y)

      labels = loadmat('../../Datasets/cifar-100/cifar-100-matlab/test.mat')['fine_labels']
      test_Y = np.zeros([labels.shape[0],100])
      for i,j in enumerate(labels):
        test_Y[i][j] = 1
      np.save('test_Y.npy', test_Y)

    train_X = loadmat('../../Datasets/cifar-100/cifar-100-matlab/train.mat')['data']
    test_X = loadmat('../../Datasets/cifar-100/cifar-100-matlab/test.mat')['data']
    
    if FLAGS.use_grayscale:
      train_X = np.array([np.reshape((p[0,:,:]*0.3 + p[1,:,:]*0.59 + p[2,:,:]*0.11), [32*32]) for p in np.reshape(train_X,[-1,3,32,32])], dtype=float)
      test_X = np.array([np.reshape((p[0,:,:]*0.3 + p[1,:,:]*0.59 + p[2,:,:]*0.11), [32*32]) for p in np.reshape(test_X,[-1,3,32,32])], dtype=float)

    for epoch_i in range(n_epochs):
        # Training
        train_accuracy = 0
        for batch_i in range(train_X.shape[0] // batch_size):
            batch_xs, batch_ys = get_cifar100(train_X, train_Y, batch_size, batch_i)
            train_accuracy += sess.run([optimizer, accuracy], feed_dict={
                x: batch_xs, y: batch_ys})[1]
        train_accuracy /= (train_X.shape[0] // batch_size)

        # Validation
        valid_accuracy = 0
        for batch_i in range(np.shape(test_X)[0] // batch_size):
            batch_xs, batch_ys = get_cifar100(test_X, test_Y, batch_size, batch_i)
            valid_accuracy += sess.run(accuracy,
                                       feed_dict={
                                           x: batch_xs,
                                           y: batch_ys
                                       })
        valid_accuracy /= (test_X.shape[0] // batch_size)
        print('epoch:', epoch_i, ', train:',
              train_accuracy, ', valid:', valid_accuracy)


if __name__ == '__main__':
    test()