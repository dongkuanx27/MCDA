#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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



def RNN(X, lstm_cell, weights, biases, instance_length, stepwidth, n_hidden_units, n_dim, num_stacked_layers, w, V, U):


    length = X.get_shape()[0]

    x_list = tf.split(X, length, 0)

    ins_list = []
    K = 0
    for i in range(0, length):
        data = tf.squeeze(x_list[i])
        data_length = data.get_shape()[0]
        K = 0
        for j in range(0, data_length - instance_length + 1, stepwidth):
            instance = data[j:j + instance_length]
            ins_list.append(instance)
            K += 1


    bt_ins = tf.convert_to_tensor(ins_list)
    init_state = lstm_cell.zero_state(len(ins_list), dtype=tf.float64)

    h_list_tensor = LSTM(bt_ins, weights, biases, lstm_cell, init_state, n_dim, instance_length, \
                         n_hidden_units, num_stacked_layers)


    res = attention(w, V, U, h_list_tensor, length, K)

    return res, h_list_tensor, K

def cap_loss(h_list_tensor, K):
    cap_loss_v = []
    binary_feature = tf.sigmoid(h_list_tensor)
    cap_loss = binary_feature[:-1] - binary_feature[1:]
    for i in range(0, cap_loss.get_shape()[0], K):
        dist_block = cap_loss[i: i + K - 1]
        hinge = tf.maximum(tf.cast(0., tf.float64), dist_block)
        cap_loss_v.append(tf.reduce_sum(hinge))
    return tf.reduce_mean(tf.convert_to_tensor(cap_loss_v))

def aps_loss(feature1, feature2):
    hinge = tf.maximum(tf.cast(0., tf.float64), tf.sigmoid(feature1)- tf.sigmoid(feature2))
    return tf.reduce_mean(tf.reduce_sum(hinge, 1))

def attention(w, V, U, h_list_tensor, length, K):
    z_bt = []
    # A = tf.transpose(tf.matmul(tf.transpose(w), tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor)))))

    # A= tf.transpose(tf.nn.softmax(tf.matmul(tf.transpose(w), tf.multiply(tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor))), \
    #                     tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    for i in range(0, length):
        h_block = h_list_tensor[i * K:(i + 1) * K]
        ##### max, mean, attention1, attention2

        # z = tf.reduce_max(h_block, axis=0)
        z = tf.reduce_mean(h_block, axis=0)

        # oneblock = A[i * K:(i + 1) * K]

        # a = tf.nn.softmax(oneblock)


        # z = tf.reduce_sum(tf.matmul(tf.transpose(a), h_block), 0)

        z_bt.append(z)
    return tf.convert_to_tensor(z_bt, dtype=tf.float64)

def LSTM(X, weights, biases, cell, init_state, n_dim, n_steps, n_hidden_units, num_stacked_layers):
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_dim])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    #
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    feature = final_state[num_stacked_layers-1][1]


    return feature



if __name__ == '__main__':

    filename = 'xxxxxx'

    num_of_sample = len(["" for line in open(filename, "r")])

    n_dim = get_dimension(filename) - 1

    data = loadData(filename, num_of_sample)


    n_hidden_units = 50
    n_classes = 2
    batch_size = 450
    n_steps = 200
    instance_length = 50
    instance_step_width = 10
    lr = 0.001  # learning rate
    training_iters = 600  # train step
    stepwidth = 100
    L = 20

    # num of stacked lstm layers
    num_stacked_layers = 1

    in_keep_prob = 0.5
    out_keep_prob = 1
    lambda_l2_reg = 0.001

    tf.reset_default_graph()
    # 对 weights biases 初始值的定义
    # weights = {
    #     # shape (28, 128)
    #     'in': tf.Variable(tf.random_normal([n_dim, n_hidden_units], dtype=tf.float64)),
    #     # shape (128, 10)
    #     'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes], dtype=tf.float64))
    # }
    # biases = {
    #     # shape (128, )
    #     'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ], dtype=tf.float64)),
    #     # shape (10, )
    #     'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ], dtype=tf.float64))
    # }
    weights = {
        'in': tf.get_variable('Weights_in', \
                              shape=[n_dim, n_hidden_units], \
                              dtype=tf.float64, \
                              initializer=tf.truncated_normal_initializer()),
        'out': tf.get_variable('Weights_out', \
                               shape=[n_hidden_units, n_classes], \
                               dtype=tf.float64, \
                               initializer=tf.truncated_normal_initializer()),
    }
    biases = {
        'in': tf.get_variable('Biases_in', \
                              shape=[n_hidden_units, ], \
                              dtype=tf.float64, \
                              initializer=tf.constant_initializer(0.)),
        'out': tf.get_variable('Biases_out', \
                               shape=[n_classes, ], \
                               dtype=tf.float64, \
                               initializer=tf.constant_initializer(0.)),
    }

    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    cells = []
    for i in range(num_stacked_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cell = tf.contrib.rnn.LSTMCell(n_hidden_units, use_peepholes=True)
            cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob,
                                                         output_keep_prob=out_keep_prob)
            cells.append(cell_dropout)
    lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)


    M = n_hidden_units
    w = tf.Variable(tf.constant(0.1, shape=[L, 1], dtype=tf.float64))
    V = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))
    U = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))

    # x y placeholder
    x = tf.placeholder(tf.float64, [batch_size, n_steps, n_dim])
    y = tf.placeholder(tf.float64, [batch_size, n_classes])

    x2 = tf.placeholder(tf.float64, [batch_size, n_steps, n_dim])

    feature, h_list_tensor, K = RNN(x, lstm_cell, weights, biases, instance_length, instance_step_width, n_hidden_units, \
                                    n_dim, num_stacked_layers,  w, V, U)
    feature2, _, _ = RNN(x2, lstm_cell, weights, biases, instance_length, instance_step_width, n_hidden_units, \
                                    n_dim, num_stacked_layers, w, V, U)

    pred = tf.matmul(feature, weights['out']) + biases['out']
    caploss = cap_loss(h_list_tensor, K)
    apsloss = aps_loss(feature, feature2)

    labelloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = labelloss + caploss + 100*apsloss
    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
            reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))
    cost += reg_loss

    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

    # init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
    # 替换成下面的写法:
    init = tf.global_variables_initializer()

    start = 1
    end = 100000
    case_id = [372589, 160302]
    label_v = [1.0, 0.0]
    tr_data, tr_y = data.generateList(n_steps, stepwidth, start, end, label_v)
    # generateBatch(n_steps, batch_size, stepwidth, start, end, label_v)

    label_v = [0.0, 1.0]
    listNum = len(tr_data)
    tr_data_case, tr_y_case = data.generateRepeatList((int)(listNum / 2), case_id[0] - n_steps, case_id[0], label_v)
    batch_xs = merge_list(tr_data, tr_data_case)
    batch_ys = merge_list(tr_y, tr_y_case)

    tr_data_case_ab, tr_y_case_ab = data.generateRepeatList((int)(listNum / 2), case_id[0], case_id[0] + n_steps, label_v)
    batch_x2 = merge_list(tr_data, tr_data_case_ab)

    tr_data_case, tr_y_case = data.generateRepeatList((int)(listNum / 2), case_id[1] - n_steps, case_id[1], label_v)

    batch_xs = merge_list(batch_xs, tr_data_case)
    batch_ys = merge_list(batch_ys, tr_y_case)

    tr_data_case_ab, tr_y_case_ab = data.generateRepeatList((int)(listNum / 2), case_id[1], case_id[1] + n_steps, label_v)
    batch_x2 = merge_list(batch_x2, tr_data_case_ab)




    testing = data.generateTest(n_steps, batch_size, stepwidth)


    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        while epoch < training_iters:
            training_accuracy = 0
            batch_num = int(len(batch_xs)/batch_size)
            idx = np.random.permutation(len(batch_xs))
            xx = np.array(batch_xs)[idx]
            yy = np.array(batch_ys)[idx]
            xx2 = np.array(batch_x2)[idx]

            for i in range(0, batch_num):
                bt = xx[i*batch_size:(i+1)*batch_size]
                by = yy[i*batch_size:(i+1)*batch_size]
                bx2 = xx2[i * batch_size:(i + 1) * batch_size]
                _, ac, lloss, closs, aloss = sess.run([train_op, accuracy, labelloss, caploss, apsloss], feed_dict={x: bt, y: by, \
                                                                                                     x2: bx2})
                # ac = accuracy.eval(feed_dict={x: bt, y: by})
                training_accuracy += ac
                #print(lloss, "  ", closs, " ", aloss)
            training_accuracy /= batch_num

            if epoch % 2 == 0:
                print(training_accuracy)
            epoch += 1
        predict=[]

        p_label = tf.argmax(pred, 1)
        for bt in testing:
            pred_label = p_label.eval(feed_dict={x: bt})
            predict= np.concatenate((predict, pred_label), axis=0)

        print(predict.tolist())