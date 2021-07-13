#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf


from LoadData import LoadData as loadData
from mill import mill as millData
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

        z = tf.reduce_max(h_block, axis=0)
        #z = tf.reduce_mean(h_block, axis=0)

        # oneblock = A[i * K:(i + 1) * K]
        #
        # a = tf.nn.softmax(oneblock)
        #
        #
        # z = tf.reduce_sum(tf.matmul(tf.transpose(a), h_block), 0)

        z_bt.append(z)
    return tf.convert_to_tensor(z_bt, dtype=tf.float64)

def LSTM(X, weights, biases, cell, init_state, n_dim, n_steps, n_hidden_units, num_stacked_layers):
    # # X ==> (128 batches * 28 steps, 28 inputs)
    # X = tf.reshape(X, [-1, n_dim])
    #
    # # X_in = W*X + b
    # X_in = tf.matmul(X, weights['in']) + biases['in']
    # # X_in ==> (128 batches, 28 steps, 128 hidden)
    # X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
    feature = final_state[num_stacked_layers-1][1]


    return feature



if __name__ == '__main__':
    n_steps = 4500#9000
    stride = 20

    n_hidden_units = 100
    n_classes = 16
    batch_size = 20
    instance_length = 100
    instance_step_width = 50
    lr = 0.001  # learning rate
    training_iters = 600  # train step
    stepwidth = 100
    n_dim = 6
    L = 50

    model_path = "/tmp/mill_model.ckpt"

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
    cost = labelloss #+ 100*apsloss #caploss # +
    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
            reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))
    #cost += reg_loss

    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

    # init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
    # 替换成下面的写法:
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    path = 'xxxxxx'
    milldata = millData(path, n_steps, stride)
    batch_xs, batch_ys = milldata.getData()

    #   test_bt_list_x, test_bt_list_y = milldata.generateBatchList(batch_size)

    tr_idx = []
    te_idx = []

    for i in range(0, n_classes):
        indSet = np.where(np.array(batch_ys)[:, i] == 1)[0]
        indices = np.random.permutation(len(indSet))

        training_idx, test_idx = indices[:len(indSet) / 2 + 1], indices[len(indSet) / 2 + 1:]
        training_idx = indSet[training_idx]
        test_idx = indSet[test_idx]
        tr_idx = merge_list(tr_idx, training_idx.tolist())
        te_idx = merge_list(te_idx, test_idx.tolist())

    training_data, test_data = np.array(batch_xs)[tr_idx], np.array(batch_xs)[te_idx]

    training_label, test_label = np.array(batch_ys)[tr_idx], np.array(batch_ys)[te_idx]

    training_bt = []
    training_label_bt = []
    testing_bt = []
    testing_label_bt = []

    batch_num = int(len(training_data) / batch_size)

    for i in range(0, batch_num):
        bt = training_data[i * batch_size:(i + 1) * batch_size]
        by = training_label[i * batch_size:(i + 1) * batch_size]
        training_bt.append(bt)
        training_label_bt.append(by)

    batch_num = int(len(test_data) / batch_size)

    for i in range(0, batch_num):
        bt = test_data[i * batch_size:(i + 1) * batch_size]
        by = test_label[i * batch_size:(i + 1) * batch_size]
        testing_bt.append(bt)
        testing_label_bt.append(by)

    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        while epoch < training_iters:
            training_accuracy = 0

            for i in range(0, len(training_bt)):
                bt = training_bt[i]
                by = training_label_bt[i]
                _, ac = sess.run([train_op, accuracy], feed_dict={x: bt[:, 0:n_steps, :], y: by, x2: bt[:, \
                                                                                                       n_steps:, :]})
                training_accuracy += ac
            training_accuracy /= len(training_bt)

            if epoch % 2 == 0:
                print(training_accuracy)
            epoch += 1

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

    # 运行一个新的会话（session）
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # 初始化变量（variables）
        sess.run(init)

        # 恢复先前保存模型的权重（weights）
        load_path = saver.restore(sess, model_path)

        test_accuracy = 0

        for i in range(0, len(testing_bt)):
            bt = testing_bt[i]
            by = testing_label_bt[i]
            ac = sess.run(accuracy, feed_dict={x: bt[:, 0:n_steps, :], y: by})
            test_accuracy += ac
        test_accuracy /= len(training_bt)
        print(test_accuracy)