import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import os, glob

from os import listdir
from os.path import isfile, join

from scipy import ndimage 
import re
 
if(os.path.exists('save') and os.path.exists('out')):
    with open('save','r') as f1:
        with open('out','r') as f2:
                if(f1 and f2):
                    f1.seek(0)
                    f2.seek(0)
                    inp=np.load(f1)
                    out=np.load(f2)
else:
    mypath="/Users/hans/Downloads/IIT/Datasets/mit_8_images"
    classes = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
    classes.remove('TEST')
    out,inp,j=[],[],0
    for i in classes:
        for f in listdir(os.path.join(mypath,i)):
            if re.match(r'.*\.jpg', f):
                inp.append(ndimage.imread(os.path.join(mypath,i,f),flatten=True))
                out.append(np.zeros(8))
                out[-1][classes.index(i)]=1
                j+=1
    inp=np.array(inp,dtype='float')
    
    print np.sum(out,axis=1),j
    
    for i in range(np.shape(inp)[0]):
        for j in range(256):
            for k in range(256):
                inp[i,j,k]=inp[i,j,k]/255         
    
    with open('save','w+') as s:
        np.save(s,inp)
    with open('out','w+') as s:
        np.save(s,out)
    
if(os.path.exists('test_save') and os.path.exists('test_out')):
    with open('test_save','r') as f1:
        with open('test_out','r') as f2:
                if(f1 and f2):
                    f1.seek(0)
                    f2.seek(0)
                    inp_test=np.load(f1)
                    out_test=np.load(f2)
else:
    mypath="/Users/hans/Downloads/IIT/Datasets/mit_8_images/TEST"
    classes = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
    
    out_test,inp_test=[],[]
    for i in classes:
        for f in listdir(os.path.join(mypath,i)):
            if re.match(r'.*\.jpg', f):
                inp_test.append(ndimage.imread(os.path.join(mypath,i,f),flatten=True))
                out_test.append([int(x) for x in '{0:03b}'.format(classes.index(i))])
        
    inp_test=np.array(inp_test,dtype='float')
    out_test=np.array(out_test,dtype='float')
    inp_test=inp_test/255
    
    with open('test_save','w+') as s:
        np.save(s,inp_test)
    with open('test_out','w+') as s:
        np.save(s,out_test)

W = tf.Variable(tf.random_normal([256,8], stddev=0.01)) 
B = tf.Variable(tf.zeros([8]))

x = tf.placeholder(tf.float32, [None,256,256])
y = tf.placeholder(tf.float32, [None,8])


def RNN(x, weight, bias):
    
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 256])
    x = tf.split(0, 256, x)

    lstm = rnn.rnn_cell.BasicLSTMCell(256,forget_bias=1.0,state_is_tuple=True)
    outputs, states = rnn.rnn(lstm,x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

res = RNN(x,W,B)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(res, y))
train_op = tf.train.AdamOptimizer(0.1).minimize(cost)

pred = tf.equal(tf.argmax(res,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

gpu = raw_input("Access GPU for computation? (y/n): ")

if(gpu=='y' or gpu=='Y'):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        step = 1
        # Keep training until reach max iterations
        while step * 2600 < 1000000:
            sess.run(train_op, feed_dict={x: inp, y: out})
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: inp, y: out})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: inp, y: out})
            print "Iter " + str(step*2600) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"
        
        print "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: inp_test, y: out_test})
    