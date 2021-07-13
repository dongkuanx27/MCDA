
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES" ] ="0"
import tensorflow as tf



from LoadData import LoadData as loadData
from TripletGenerator import TripletGenerator as batchPool

import numpy as np

def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2



def attention(h_list_tensor, K):
    #A = tf.transpose(tf.matmul(tf.transpose(w), tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor)))))

    A= tf.transpose(tf.nn.softmax(tf.matmul(tf.transpose(w), tf.multiply(tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor))), \
                         tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    dim = h_list_tensor.get_shape()[1].value


    h_block_list = tf.reshape(h_list_tensor, [-1, K, dim])


    A_list = tf.reshape(A, [-1, K])

    #z_bt = tf.reduce_max(h_block_list, axis=1)
    #z_bt = tf.reduce_mean(h_block_list, axis=1)


    a_list = tf.nn.softmax(A_list, dim=1)

    a_list = tf.expand_dims(a_list, 1)


    z_bt = tf.matmul(a_list, h_block_list)

    if (len(z_bt.get_shape()) > 2):
        return tf.squeeze(z_bt, squeeze_dims=1)
    else:
        return z_bt


def mi_lstm(lstm_cell, X, instance_length, stepwidth, n_hidden_units, n_dim, num_stacked_layers):
    K = 0
    ins_list = []
    data_length = X.get_shape()[1]
    for j in range(0, data_length - instance_length + 1, stepwidth):
        instance = X[:, j:j + instance_length, :]
        ins_list.append(instance)
        K += 1

    ins_list = tf.transpose(ins_list, [1, 0, 2, 3])
    # bt_ins = tf.concat(ins_list, axis=0)
    bt_ins = tf.reshape(ins_list, [-1, n_hidden_units, n_dim])

    # init_state = lstm_cell.zero_state(bt_ins.get_shape()[0].value, dtype=tf.float64)

    h_list_tensor = LSTM(bt_ins, lstm_cell, n_dim, instance_length, \
                         n_hidden_units, num_stacked_layers)
    res = attention(h_list_tensor, K)
    return res, h_list_tensor, K


def LSTM(X, cell, n_dim, n_steps, n_hidden_units, num_stacked_layers):
    # # X ==> (128 batches * 28 steps, 28 inputs)
    # X = tf.reshape(X, [-1, n_dim])
    #
    # # X_in = W*X + b
    # X_in = tf.matmul(X, weights['in']) + biases['in']
    # # X_in ==> (128 batches, 28 steps, 128 hidden)
    # X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # #
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, time_major=False)


    feature = final_state[num_stacked_layers - 1][1]


    return feature

def aps_loss(feature1, feature2):
    hinge = tf.maximum(tf.cast(0., tf.float64), tf.sigmoid(feature1)- tf.sigmoid(feature2))
    return tf.reduce_sum(hinge, 1)

def cap_loss(h_list_tensor, K):
    binary_feature = tf.sigmoid(h_list_tensor)
    cap_loss = binary_feature[:-1] - binary_feature[1:0]
    # for i in range(0, cap_loss.get_shape()[0], K):
    #     dist_block = cap_loss[i: i + K - 1]
    #     hinge = tf.maximum(tf.cast(0., tf.float64), dist_block)
    #     cap_loss_v.append(tf.reduce_sum(hinge))


    # size of x dimension
    # random roll amount

    binary_feature_shifted = tf.manip.roll(binary_feature, shift=[1, 0], axis=[0, 1])

    cap_loss = binary_feature - binary_feature_shifted

    dim = cap_loss.get_shape()[1].value
    dist_block = tf.reshape(cap_loss, [-1, K, dim])
    dist_block = dist_block[:, 0:K-1, :]
    hinge = tf.maximum(tf.cast(0., tf.float64), dist_block)
    loss = tf.reduce_sum(tf.reduce_sum(hinge, axis=1), axis=1)

    return loss

def distance(z1, z2):
    ## l2diff
    #return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z1, z2)), reduction_indices=1))
    return tf.sqrt(tf.reduce_sum(tf.square(z1 - z2), 1))

if __name__ == '__main__':
 
    filename = 'xxxxxx'

    num_of_sample = len(["" for line in open(filename, "r")])

    n_dim = get_dimension(filename) - 1

    data = loadData(filename, num_of_sample)

    n_hidden_units = 50
    n_classes = 2
    batch_size = 200
    n_steps = 200
    lr = 0.001  # learning rate
    training_iters = 500  # train step
    stepwidth = 100
    margin = 10

    # num of stacked lstm layers
    num_stacked_layers = 1

    instance_length = 50
    instance_step_width = 10
    L = 20


    in_keep_prob = 0.5
    out_keep_prob = 1
    lambda_l2_reg = 0.001


    tf.reset_default_graph()

    # weights = {
    #     # shape (28, 128)
    #     'in': tf.Variable(tf.random_normal([n_dim, n_hidden_units], dtype=tf.float64)),
    # }
    # biases = {
    #     # shape (128, )
    #     'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ], dtype=tf.float64)),
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
    w = tf.Variable(name='Weights_w', initial_value=tf.constant(0.1, shape=[L, 1], dtype=tf.float64))
    V = tf.Variable(name='Weights_V', initial_value=tf.constant(0.1, shape=[L, M], dtype=tf.float64))
    U = tf.Variable(name='Weights_U', initial_value=tf.constant(0.1, shape=[L, M], dtype=tf.float64))

    # x placeholder
    x1 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    x2 = tf.placeholder(tf.float64, [None, n_steps, n_dim])

    x3 = tf.placeholder(tf.float64, [None, n_steps, n_dim])

    y1 = tf.placeholder(tf.float64, [None, ])
    y2 = tf.placeholder(tf.float64, [None, ])
    y3 = tf.placeholder(tf.float64, [None, ])

    # x placeholder
    x11 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    x22 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    x33 = tf.placeholder(tf.float64, [None, n_steps, n_dim])

    start = 1
    end = 100000
    case_id = [372589, 160302]  # 513081
    label_v = [1.0, 0.0]
    negative, tr_y = data.generateList(n_steps, stepwidth, start, end, label_v)

    listNum = len(negative)
    label_v = [0.0, 1.0]
    anchor, tr_y_case1 = data.generateRepeatList(listNum, case_id[0] - n_steps, case_id[0], label_v)
    positive, _ = data.generateRepeatList(listNum, case_id[1] - n_steps, case_id[1], label_v)
    anchor2, tr_y_case1 = data.generateRepeatList(listNum, case_id[0], case_id[0] + n_steps, label_v)
    pos2, _ = data.generateRepeatList(listNum, case_id[1], case_id[1] + n_steps, label_v)

    aps_cap_flag = True
    batch_data_pool = batchPool(aps_cap_flag)

    batch_data_pool.addTriplet_apscap(anchor, positive, negative, anchor2, [1] * listNum, pos2, [1] * listNum, negative, [0] * listNum)

    testing = data.generateTest(n_steps, batch_size, stepwidth)

    anchor_output, h_list_tensor1, K1 = mi_lstm(lstm_cell, x1, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    anchor_output2, _, _ = mi_lstm(lstm_cell, x11, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    positive_output, h_list_tensor2, K2 = mi_lstm(lstm_cell, x2, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    positive_output2, _, _ = mi_lstm(lstm_cell, x22, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    negative_output, h_list_tensor3, K3 = mi_lstm(lstm_cell, x3, instance_length, instance_step_width, n_hidden_units, n_dim,
                                     num_stacked_layers)
    negative_output2, _, _ = mi_lstm(lstm_cell, x33, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)



    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(tf.cast(0., tf.float64), margin + d_pos - d_neg)
    trloss = tf.reduce_mean(loss)

    apsloss = tf.reduce_mean(y1* aps_loss(anchor_output, anchor_output2) + y2*aps_loss(positive_output, positive_output2) \
                             + y3*aps_loss(negative_output, negative_output2))

    caploss = tf.reduce_mean(y1* cap_loss(h_list_tensor1, K1)) + tf.reduce_mean(y2* cap_loss(h_list_tensor2, K2)) \
              + tf.reduce_mean(y3* cap_loss(h_list_tensor3, K3))

    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
            reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))


    cost = trloss + apsloss + caploss + 0.01*reg_loss


    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # gvs = optimizer.compute_gradients(cost)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # train_op = optimizer.apply_gradients(capped_gvs)

    init = tf.global_variables_initializer()





    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        while epoch < training_iters:
            training_loss = 0
            tr_loss = 0
            aps = 0
            cap = 0
            reg = 0
            stepNum = (int)(batch_data_pool.getTrainNum()/batch_size)
            step = 0
           # batch_data_pool.reshuffle()
            while batch_data_pool.hasNext():
                bt1, bt2, bt3, bt11, bt22, bt33, bt_tag1, bt_tag2, bt_tag3 = batch_data_pool.next_batch(batch_size)

                _, loss, triplet_loss, capLS, apsLS, regloss = sess.run([train_op, cost, trloss, \
                                                                         caploss, apsloss, reg_loss], feed_dict={x1: bt1, x2: bt2, x3: bt3, x11: bt11, x22: bt22, x33: bt33, y1: bt_tag1, y2: bt_tag2, y3: bt_tag3})
                training_loss += loss
                tr_loss += triplet_loss
                aps += apsLS
                cap += capLS
                reg += regloss
                step += 1


            training_loss_avg = training_loss / (batch_data_pool.getTrainNum())
            tr_loss = tr_loss / (batch_data_pool.getTrainNum())
            aps = aps / ((step + 1) * batch_size)
            cap = cap / ((step + 1) * batch_size)
            print("loss:", training_loss_avg, " triplet loss: ", tr_loss, " cap loss:", cap, " aps loss:", aps, \
                  "reg loss:", regloss)
            epoch += 1
        predict = []

        test_case = anchor[0: batch_size]

        dist_z1_z2_test = distance(anchor_output, positive_output)
        for bt in testing:
            dist_test = dist_z1_z2_test.eval(feed_dict={x1: test_case, x2: bt})
            predict = np.concatenate((predict, dist_test), axis=0)

        print(predict.tolist())