
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES" ] ="1"
import tensorflow as tf



from PairGenerator import PairGenerator as batchPool
from mill import mill as millData
import copy

import numpy as np
from tensorflow.python.client import device_lib

def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2

def RNN(X1, X2, instance_length, stepwidth, n_hidden_units, n_dim, num_stacked_layers, in_keep_prob, out_keep_prob):
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    cells = []
    for i in range(num_stacked_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cell = tf.contrib.rnn.LSTMCell(n_hidden_units, use_peepholes=True)
            cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob,
                                                         output_keep_prob=out_keep_prob)
            cells.append(cell_dropout)
    lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)


    feature1 = mi_lstm(lstm_cell, X1, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)

    feature2 = mi_lstm(lstm_cell, X2, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)

    return feature1, feature2


def attention(h_list_tensor, K):
    A = tf.transpose(tf.matmul(tf.transpose(w), tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor)))))

    #A= tf.transpose(tf.nn.softmax(tf.matmul(tf.transpose(w), tf.multiply(tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor))), \
    #                     tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    dim = h_list_tensor.get_shape()[1].value


    h_block_list = tf.reshape(h_list_tensor, [-1, K, dim])




    z_bt = tf.reduce_max(h_block_list, axis=1)
    #z_bt = tf.reduce_mean(h_block_list, axis=1)

    # A_list = tf.reshape(A, [-1, K])
    # a_list = tf.nn.softmax(A_list, dim=1)
    #
    # a_list = tf.expand_dims(a_list, 1)
    #
    #
    # z_bt = tf.matmul(a_list, h_block_list)

    if(len(z_bt.get_shape()) > 2):
        return tf.squeeze(z_bt, squeeze_dims=1)
    else:
        return z_bt


def mi_lstm(lstm_cell, X, instance_length, stepwidth, n_hidden_units, n_dim, num_stacked_layers):
    # length = X.get_shape()[0]
    #
    # x_list = tf.split(X, length, 0)
    #
    # ins_list = []
    # K =0
    # for i in range(0, length):
    #     data = tf.squeeze(x_list[i])
    #     data_length = data.get_shape()[0]
    #     K = 0
    #     for j in range(0, data_length - instance_length + 1, stepwidth):
    #         instance = data[j:j + instance_length]
    #         ins_list.append(instance)
    #         K += 1
    # bt_ins = tf.convert_to_tensor(ins_list)


    K = 0
    ins_list = []
    data_length = X.get_shape()[1]
    for j in range(0, data_length - instance_length + 1, stepwidth):
        instance = X[:, j:j + instance_length, :]
        ins_list.append(instance)
        K += 1
    #bt_ins = tf.concat(ins_list, axis=0)

    ins_list = tf.transpose(ins_list, [1, 0, 2, 3])
    bt_ins = tf.reshape(ins_list, [-1, instance_length, n_dim])

    #init_state = lstm_cell.zero_state(len(ins_list), dtype=tf.float64)

    h_list_tensor = LSTM(bt_ins, lstm_cell, n_dim, instance_length, \
                         n_hidden_units, num_stacked_layers)
    res = attention(h_list_tensor, K)
    return res


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


def distance(z1, z2):
    ## l2diff
    #return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z1, z2)), reduction_indices=1))
    return tf.sqrt(tf.reduce_sum(tf.square(z1 - z2), 1)+1e-30)

def normalize(v):
    return tf.nn.l2_normalize(v, axis=1, epsilon=1e-12)

def shiftOne(arr):
    tmp = arr[0]
    for i in range(0, len(arr)-1):
        arr[i] = arr[i + 1]
    arr[len(arr)-1] = tmp
    return arr


if __name__ == '__main__':
    n_steps = 9000
    stride = 20
    n_hidden_units = 30
    n_classes = 16
    batch_size = 10
    instance_length = 100
    instance_step_width = 100
    lr = 0.01  # learning rate
    training_iters = 50  # train step
    n_dim = 6
    L = 30
    margin = 1

    # num of stacked lstm layers
    num_stacked_layers = 1
    model_path = "./mill_model.ckpt"


    in_keep_prob = 1#0.5
    out_keep_prob = 1
    lambda_l2_reg = 0.001


    tf.reset_default_graph()

    print(device_lib.list_local_devices())

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

    M = n_hidden_units
    w = tf.Variable(tf.constant(0.1, shape=[L, 1], dtype=tf.float64))
    V = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))
    U = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))

    # x placeholder
    x1 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    x2 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
    y = tf.placeholder(tf.float64, [None,])

    y_tr = tf.placeholder(tf.float64, [None, n_classes])

    z1, z2 = RNN(x1, x2, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers, in_keep_prob, out_keep_prob)


    z1 = normalize(z1)
    z2 = normalize(z2)


    dist_z1_z2 = distance(z1, z2)
    hingeloss = tf.maximum(tf.cast(0., tf.float64), margin - dist_z1_z2)
    dissimilarloss = tf.reduce_mean((1 - y) * tf.pow(dist_z1_z2, 2))
    similarloss = tf.reduce_mean(y * tf.pow(hingeloss, 2))
    cost = tf.reduce_mean((1 - y) * tf.pow(dist_z1_z2, 2) + y * tf.pow(hingeloss, 2))

    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
            reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))
    #cost += reg_loss

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

    saver = tf.train.Saver()

    path = 'xxxxxx' 
    milldata = millData(path, n_steps, stride)
    batch_xs, batch_ys = milldata.getData()

    #   test_bt_list_x, test_bt_list_y = milldata.generateBatchList(batch_size)

    tr_idx = []
    te_idx = []

    n_classes = 6#16
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



    aps_cap_flag = False
    batch_data_pool = batchPool(aps_cap_flag)

    trainData_img = copy.deepcopy(training_data)
    trainLabel_img = copy.deepcopy(training_label)


    for i in range(len(trainData_img)):
        trainData_img = shiftOne(trainData_img)
        trainLabel_img = shiftOne(trainLabel_img)
        batch_data_pool.addPair(training_data.tolist(), training_label.tolist(), trainData_img.tolist(), trainLabel_img.tolist())



    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_op)
        epoch = 0
        while epoch < training_iters:
            training_loss = 0
            disloss_a = 0
            similoss_a = 0
            stepNum = 1#(int)(batch_data_pool.getTrainNum()/batch_size)
            step = 0
           # batch_data_pool.reshuffle()
            while batch_data_pool.hasNext():
                bt1, bt2, y_label = batch_data_pool.next_batch(batch_size)

                _, loss, disloss, similoss = sess.run([train_op, cost, dissimilarloss, similarloss], feed_dict={x1: bt1, x2: bt2, y: y_label})

                training_loss += loss
                disloss_a += disloss
                similoss_a += similoss
                step += 1
         #       print('loss: ', loss)

            training_loss_avg = training_loss / (batch_data_pool.getTrainNum())

            print(epoch, ': ', training_loss_avg, 'dis loss: ', disloss_a/(batch_data_pool.getTrainNum()), \
                  'simi loss: ', similoss_a/(batch_data_pool.getTrainNum()))
            epoch += 1
        predict = []

    #    save_path = saver.save(sess, model_path)
    #    print("Model saved in file: %s" % save_path)

        #dist_z1_z2_test = distance(z1, z2)
        # for bt in testing:
        #     dist_test = dist_z1_z2_test.eval(feed_dict={x1: bt, x2: test_case})
        #     predict = np.concatenate((predict, dist_test), axis=0)
        #
        # print(predict.tolist())


        # # 恢复先前保存模型的权重（weights）
        # load_path = saver.restore(sess, model_path)


        accuracy = 0
        for i in range(len(test_data)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={x1: training_data, y_tr: training_label, x2: [test_data[i, :]]})
#            aa = sess.run(aaa, feed_dict={x1: training_data, y_tr: training_label, x2: [test_data[i, :]]})
            # Get nearest nei
            # ghbor class label and compare it to its true label
            print("Test", i, "Prediction:", nn_index,
                  "True Class:", np.argmax(test_label[i]))
            # Calculate accuracy
            if nn_index == np.argmax(test_label[i]):
                accuracy += 1. / len(test_data)
        print("Done!")
        print("Accuracy:", accuracy)