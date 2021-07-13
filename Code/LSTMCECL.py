#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf



from LoadData import LoadData as loadData
import numpy as np

def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2

def RNN(X1, X2, weights, biases, n_steps, n_hidden_units, batch_size, n_dim):
    # X ==> (128 batches * 28 steps, 28 inputs)
    X1 = tf.reshape(X1, [-1, n_dim])
    X2 = tf.reshape(X2, [-1, n_dim])

    # X_in = W*X + b
    X_in_1 = tf.matmul(X1, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden)
    X_in_1 = tf.reshape(X_in_1, [-1, n_steps, n_hidden_units])
    #
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float64) #

    outputs_1, final_state_1 = tf.nn.dynamic_rnn(lstm_cell, X_in_1, initial_state=init_state, time_major=False)

    # X_in = W*X + b
    X_in_2 = tf.matmul(X2, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden)
    X_in_2 = tf.reshape(X_in_2, [-1, n_steps, n_hidden_units])
    #
    outputs_2, final_state_2 = tf.nn.dynamic_rnn(lstm_cell, X_in_2, initial_state=init_state, time_major=False)

    return final_state_1[1], final_state_2[1]

def distance(z1, z2):
    ## l2diff
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z1, z2)), reduction_indices=1))

if __name__ == '__main__':

    filename = 'xxxxxx'

    num_of_sample = len(["" for line in open(filename, "r")])

    n_dim = get_dimension(filename) - 1

    data = loadData(filename, num_of_sample)


    n_hidden_units = 50
    n_classes = 2
    batch_size = 300
    n_steps = 200
    lr = 0.001  # learning rate
    training_iters = 200  # train step
    stepwidth = 100
    margin = 10


    # 对 weights biases 初始值的定义
    weights = {
        # shape (28, 128)
        'in': tf.Variable(tf.random_normal([n_dim, n_hidden_units], dtype=tf.float64)),
    }
    biases = {
        # shape (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ], dtype=tf.float64)),
    }

    # x placeholder
    x1 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    x2 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    y = tf.placeholder(tf.float64)


    z1, z2 = RNN(x1, x2, weights, biases, n_steps, n_hidden_units, batch_size, n_dim)
    dist_z1_z2 = distance(z1, z2)
    hingeloss = tf.maximum(tf.cast(0., tf.float64), margin - dist_z1_z2)

    cost = tf.reduce_mean((1-y) * tf.pow(dist_z1_z2, 2) + y * tf.pow(hingeloss, 2))

    #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)


    # init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
    # 替换成下面的写法:
    init = tf.global_variables_initializer()

    start = 1
    end = 100000
    case_id = 372589#160302#513081
    label_v = [1.0, 0.0]
    tr_data, tr_y = data.generateList(n_steps, stepwidth, start, end, label_v)
    # generateBatch(n_steps, batch_size, stepwidth, start, end, label_v)

    label_v = [0.0, 1.0]
    listNum = len(tr_data)
    tr_data_case, tr_y_case = data.generateRepeatList(listNum, case_id - n_steps, case_id, label_v)
    batch_xs1 = tr_data
    batch_xs2 = tr_data_case

    testing = data.generateTest(n_steps, batch_size, stepwidth)

    test_case = batch_xs2[0: batch_size]

    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        while epoch < training_iters:
            training_loss = 0
            batch_num = int(len(batch_xs1)/batch_size)
            idx = np.random.permutation(len(batch_xs1))
            xx1 = np.array(batch_xs1)[idx]
            yy1 = np.array(batch_xs1)[idx]

            idx = np.random.permutation(len(batch_xs2))
            xx2 = np.array(batch_xs2)[idx]
            yy2 = np.array(batch_xs2)[idx]

            for i in range(0, batch_num):
                bt1 = xx1[i * batch_size:(i + 1) * batch_size]
                bt2 = xx2[i * batch_size:(i + 1) * batch_size]
                sess.run([train_op], feed_dict={x1: bt1, x2: bt2, y: 1})
                loss = cost.eval(feed_dict={x1: bt1, x2: bt2, y: 1})
                training_loss += loss
            # if batch_size*batch_num < len(batch_xs):
            #     bt = xx[batch_size*batch_num:end]
            #     by = yy[batch_size*batch_num:end]
            #     sess.run([train_op], feed_dict={x: bt, y: by})
            #     print("accuracy:", accuracy.eval(feed_dict={x: bt, y: by}))
            #     training_accuracy += accuracy
            #     batch_num += 1
            training_loss /= batch_num

            if epoch % 2 == 0:
                print(training_loss)
            epoch += 1
        predict=[]

        dist_z1_z2_test = distance(z1, z2)
        for bt in testing:
            dist_test = dist_z1_z2_test.eval(feed_dict={x1: bt, x2: test_case})
            predict = np.concatenate((predict, dist_test), axis=0)

        print(predict.tolist())