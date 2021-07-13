# encoding: utf-8
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES" ] ="2"
import tensorflow as tf

#from tensorflow.python.ops import gen_manip_ops as _gen_manip_ops



from PairGenerator import PairGenerator as batchPool
from mill import mill as millData
import copy

import numpy as np

def shiftOne(arr):
    tmp = arr[0]
    for i in range(0, len(arr)-1):
        arr[i] = arr[i + 1]
    arr[len(arr)-1] = tmp
    return arr

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
    bt_ins = tf.reshape(ins_list, [-1, instance_length, n_dim])

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

    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, time_major=False)


    feature = final_state[num_stacked_layers - 1][1]


    return feature

def aps_loss(feature1, feature2):
    hinge = tf.maximum(tf.cast(0., tf.float64), tf.sigmoid(feature1)- tf.sigmoid(feature2))
    return tf.reduce_sum(hinge, 1)

def cap_loss(h_list_tensor, K):
    cap_loss_v = []
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
    return tf.sqrt(tf.reduce_sum(tf.square(z1 - z2), 1)+1e-30)

def distance2(z1, z2):
    ## l2diff
    #return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z1, z2)), reduction_indices=1))
    res = []
    length = z2.get_shape()[0]
    for i in range(0, length):
        res.append(tf.sqrt(tf.reduce_sum(tf.square(z1 - z2[i]))))
    return res

if __name__ == '__main__':
    n_steps = 9000
    stride = 20
    n_hidden_units = 30
    n_classes = 16
    batch_size = 10
    instance_length = 100
    instance_step_width = 100
    lr = 0.001  # learning rate
    training_iters = 300  # train step
    n_dim = 6
    L = 30
    margin = 1

    # num of stacked lstm layers
    num_stacked_layers = 1

    instance_length = 50
    instance_step_width = 10
    L = 20

    n_steps_tr = int(n_steps / 2)


    in_keep_prob = 0.5
    out_keep_prob = 1
    lambda_l2_reg = 0.001

    model_path = "/tmp/model.ckpt"

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
    w = tf.Variable(tf.constant(0.1, shape=[L, 1], dtype=tf.float64))
    V = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))
    U = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))

    # x placeholder
    x1 = tf.placeholder(tf.float64, [None, n_steps_tr, n_dim])
    x2 = tf.placeholder(tf.float64, [None, n_steps_tr, n_dim])
    y = tf.placeholder(tf.float64, [None,])

    y1 = tf.placeholder(tf.float64, [None, ])
    y2 = tf.placeholder(tf.float64, [None, ])

    y_tr = tf.placeholder(tf.float64, [None, n_classes])

    # x placeholder
    x11 = tf.placeholder(tf.float64, [None, n_steps_tr, n_dim])
    x22 = tf.placeholder(tf.float64, [None, n_steps_tr, n_dim])



    z1, h_list_tensor1, K1 = mi_lstm(lstm_cell, x1, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    z11, _, _ = mi_lstm(lstm_cell, x11, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    z2, h_list_tensor2, K2 = mi_lstm(lstm_cell, x2, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    z22, _, _ = mi_lstm(lstm_cell, x22, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)

    dist_z1_z2 = distance(z1, z2)
    hingeloss = tf.maximum(tf.cast(0., tf.float64), margin - dist_z1_z2)
    clloss = tf.reduce_mean((1 -y) * tf.pow(dist_z1_z2, 2) + y * tf.pow(hingeloss, 2))

    apsloss = tf.reduce_mean(y1* aps_loss(z1, z11) + y2*aps_loss(z2, z22))

    caploss = tf.reduce_mean(y1* cap_loss(h_list_tensor1, K1)) + tf.reduce_mean(y2* cap_loss(h_list_tensor2, K2))

    dissimilarloss = tf.reduce_mean((1 - y) * tf.pow(dist_z1_z2, 2))
    similarloss = tf.reduce_mean(y * tf.pow(hingeloss, 2))

    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
            reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))


    cost = clloss + apsloss   + caploss #+ 0.01*reg_loss


    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # gvs = optimizer.compute_gradients(cost)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # train_op = optimizer.apply_gradients(capped_gvs)



    K = 3
    # Euclidean Distance
    dist = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z1, z2)), reduction_indices=1)))
    # Prediction: Get min distance neighbors
    values, indi = tf.nn.top_k(dist, k=K, sorted=False)
    nearest_neighbors = []
    for i in range(K):
        nearest_neighbors.append(tf.argmax(y_tr[indi[i]], 0))

    neighbors_tensor = tf.stack(nearest_neighbors)

    yy, idx, count = tf.unique_with_counts(neighbors_tensor)

    pred = tf.slice(yy, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]



    init = tf.global_variables_initializer()

    init_op = tf.initialize_all_variables()

    path = 'xxxxxx'
    milldata = millData(path, n_steps, stride)
    batch_xs, batch_ys = milldata.getData()

    #   test_bt_list_x, test_bt_list_y = milldata.generateBatchList(batch_size)

    tr_idx = []
    te_idx = []

    n_classes = 2
    for i in range(0, n_classes):
        indSet = np.where(np.array(batch_ys)[:, i] == 1)[0]
        indices = np.random.permutation(len(indSet))

        training_idx, test_idx = indices[:int(len(indSet) / 2) + 1], indices[int(len(indSet) / 2) + 1:]
        training_idx = indSet[training_idx]
        test_idx = indSet[test_idx]
        tr_idx = merge_list(tr_idx, training_idx.tolist())
        te_idx = merge_list(te_idx, test_idx.tolist())

    training_data, test_data = np.array(batch_xs)[tr_idx], np.array(batch_xs)[te_idx]

    training_label, test_label = np.array(batch_ys)[tr_idx], np.array(batch_ys)[te_idx]



    aps_cap_flag = True
    batch_data_pool = batchPool(aps_cap_flag)

    trainData_img = copy.deepcopy(training_data)
    trainLabel_img = copy.deepcopy(training_label)


    for i in range(len(trainData_img)):
        trainData_img = shiftOne(trainData_img)
        trainLabel_img = shiftOne(trainLabel_img)
        batch_data_pool.addPair_apscap(training_data[:, 0:n_steps_tr, :].tolist(), \
                                       training_label.tolist(), trainData_img[:, 0:n_steps_tr, :].tolist(), \
                                       trainLabel_img.tolist(), training_data[:, n_steps_tr:, :].tolist(), \
                                       [1] * len(training_data), trainData_img[:, n_steps_tr:, :].tolist(), \
                                       [1] * len(trainData_img))



    # 'Saver' 操作将保存所有变量以供恢复
    saver = tf.train.Saver()

    # temp_list = []
    # for i in range(0, batch_size):
    #     ins = h_list_tensor1[i * K1:(i + 1) * K1]
    #     temp = distance2(z2[i], ins)
    #     temp_list = merge_list(temp_list, temp)
    # dist_z1_z2_test = tf.convert_to_tensor(temp_list)

    #dist_z1_z2_test = distance(z1, z2)
    #"""
    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_op)
        epoch = 0
        while epoch < training_iters:
            training_loss = 0
            cont_loss = 0
            aps = 0
            cap = 0
            reg = 0
            stepNum = (int)(batch_data_pool.getTrainNum()/batch_size)
            step = 0
            disloss_a = 0
            similoss_a = 0
           # batch_data_pool.reshuffle()
            while batch_data_pool.hasNext():
                bt1, bt2, y_label, bt11, bt22, tag1, tag2 = batch_data_pool.next_batch(batch_size)

                _, loss, contrast_loss, capLS, apsLS, dis_, sim_ = sess.run([train_op, cost, clloss, caploss, apsloss, dissimilarloss,\
                                                                 similarloss], feed_dict={x1: bt1, \
                                                                                 x2: bt2, y: y_label, x11: bt11, \
                                                                                 x22: bt22, y1: tag1, y2: tag2})
                training_loss += loss
                cont_loss += contrast_loss
                aps += apsLS
                cap += capLS
                disloss_a += dis_
                similoss_a += sim_
                step += 1
                #print("loss:", training_loss, " contrastive loss: ", cont_loss, " cap loss:", cap, " aps loss:",
                #      aps, "reg loss:", regloss)

            training_loss_avg = training_loss / (batch_data_pool.getTrainNum())
            cont_loss = cont_loss / ((step + 1) * batch_size)
            aps = aps / ((step + 1) * batch_size)
            cap = cap / ((step + 1) * batch_size)
            print("loss:", training_loss_avg, " contrastive loss: ", cont_loss, " cap loss:", cap, " aps loss:", aps, "dis: ", disloss_a/(batch_data_pool.getTrainNum()), \
                  'simi loss: ', similoss_a/(batch_data_pool.getTrainNum()))
            epoch += 1




        # 将模型的权重（weights）保存至硬盘
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

    #"""
    # 运行一个新的会话（session）
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # 初始化变量（variables）
        sess.run(init)
        sess.run(init_op)

        # 恢复先前保存模型的权重（weights）
        load_path = saver.restore(sess, model_path)
       # print("Model restored from file: %s" % save_path)

        accuracy = 0
        for i in range(len(test_data)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={x1: training_data[:, 0:n_steps_tr, :].tolist(), y_tr: training_label, x2: [test_data[i, 0:n_steps_tr, :]]})
            # Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", nn_index,
                  "True Class:", np.argmax(test_label[i]))
            # Calculate accuracy
            if nn_index == np.argmax(test_label[i]):
                accuracy += 1. / len(test_data)
        print("Done!")
        print("Accuracy:", accuracy)