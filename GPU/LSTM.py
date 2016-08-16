import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os.path

import scipy.io as sio

flag=1

if(os.path.exists('save') and os.path.exists('out') and os.path.exists('test_save') and os.path.exists('test_out')):
    with open('save','r') as f1:
        with open('out','r') as f2:
                if(f1 and f2):
                    f1.seek(0)
                    f2.seek(0)
                    inp=np.load(f1)
                    out=np.load(f2)
    
    with open('test_save','r') as f1:
        with open('test_out','r') as f2:
                if(f1 and f2):
                    f1.seek(0)
                    f2.seek(0)
                    inp_test=np.load(f1)
                    out_test=np.load(f2)
                    
else:
    
    X = sio.loadmat('Datasets/features/local/MIT8_Local_Fold1_classwise_Normalized.mat')

    inp = np.concatenate((X['train1'],np.concatenate((X['train2'],np.concatenate((X['train3'],np.concatenate((X['train4'],np.concatenate((X['train5'],np.concatenate((X['train6'],np.concatenate((X['train7'],X['train8']),axis=0)),axis=0)),axis=0)),axis=0)),axis=0)),axis=0)),axis=0)
    out = np.zeros((np.shape(X['trainDataY'])[0],8),dtype='float')
    for i,v in enumerate(X['trainDataY']):
        out[i][v[0]-1]=1
    print np.shape(inp)
    inp = np.reshape(inp,[np.shape(inp)[0]/36,36,22])

    inp_test = np.reshape(X['testDataX'],[np.shape(X['testDataX'])[0],36,22])
    out_test = np.zeros((np.shape(inp_test)[0],8),dtype='float')
    for k,v in enumerate(X['testDataY']):
        out_test[k][v[0]-1]=1

    with open('save','w+') as s:
        np.save(s,inp)
    with open('out','w+') as s:
        np.save(s,out)
    
    with open('test_save','w+') as s:
        np.save(s,inp_test)
    with open('test_out','w+') as s:
        np.save(s,out_test)

x = tf.placeholder(tf.float32, [None,36,22])
y = tf.placeholder(tf.float32, [None,8])

W = tf.Variable(tf.random_normal([200,8], stddev=0.01))
W_in = tf.Variable(tf.random_normal([22,200], stddev=0.01))
B_in = tf.Variable(tf.zeros([200]))
B = tf.Variable(tf.zeros([8]))

def RNN(x, weight, bias):
    
    if(flag==1): 
        x = tf.reshape(tf.matmul(tf.reshape(x,[np.shape(inp)[0]*36,22]),W_in),[int(np.shape(inp)[0]),36,200])+B_in
    else:
        x = tf.reshape(tf.matmul(tf.reshape(x,[np.shape(inp_test)[0]*36,22]),W_in),[int(np.shape(inp_test)[0]),36,200])+B_in
    
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 200])
    x = tf.split(0, 36, x)

    lstm = rnn.rnn_cell.BasicLSTMCell(200,forget_bias=1.0,state_is_tuple=True)
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
        while step * 1200 < 100000:
            sess.run(train_op, feed_dict={x: inp, y: out})
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: inp, y: out})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: inp, y: out})
            print "Iter " + str(step*1200) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"

        flag=0
        print "Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: inp_test, y: out_test})
