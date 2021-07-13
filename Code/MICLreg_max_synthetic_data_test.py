
import sys
import os
import random
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID"   # see issue #152
import tensorflow as tf
import xml.etree.ElementTree as ET


from LoadData3 import LoadData as loadData
from PairGenerator2 import PairGenerator as batchPool
from tssutil import TSSUtil
from dataElement import DataElement
import numpy as np
import pandas as pd

def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2



def attention(h_list_tensor, length, K):
    z_bt = []
    #A = tf.transpose(tf.matmul(tf.transpose(w), tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor)))))

    A= tf.transpose(tf.nn.softmax(tf.matmul(tf.transpose(w), tf.multiply(tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor))), \
                          tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    for i in range(0, length):
        h_block = h_list_tensor[i * K:(i + 1) * K]
        ##### max, mean, attention1, attention2

        z = tf.reduce_max(h_block, axis=0)
        #z = tf.reduce_mean(h_block, axis=0)

        # oneblock = A[i * K:(i + 1) * K]
        # a = tf.nn.softmax(oneblock)
        # z = tf.reduce_sum(tf.matmul(tf.transpose(a), h_block), 0)


        z_bt.append(z)
    return tf.convert_to_tensor(z_bt, dtype=tf.float64)


def mi_lstm(lstm_cell, X, instance_length, stepwidth, n_hidden_units, n_dim, num_stacked_layers):
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

    h_list_tensor = LSTM(bt_ins, lstm_cell, init_state, n_dim, instance_length, \
                         n_hidden_units, num_stacked_layers)
    res = attention(h_list_tensor, length, K)
    return res, h_list_tensor, K


def LSTM(X, cell, init_state, n_dim, n_steps, n_hidden_units, num_stacked_layers):
    # X ==> (128 batches * 28 steps, 28 inputs)
 #   X = tf.reshape(X, [-1, n_dim])

    # X_in = W*X + b
 #   X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden)
    X_in = X#tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    #
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)


    feature = final_state[num_stacked_layers - 1][1]


    return feature


def aps_loss(feature1, feature2):
    hinge = tf.maximum(tf.cast(0., tf.float64), tf.sigmoid(feature1)- tf.sigmoid(feature2))
    return tf.reduce_sum(hinge, 1)

def cap_loss(h_list_tensor, K):
    cap_loss_v = []
    binary_feature = tf.sigmoid(h_list_tensor)
    cap_loss = binary_feature[:-1] - binary_feature[1:]
    for i in range(0, cap_loss.get_shape()[0], K):
        dist_block = cap_loss[i: i + K - 1]
        hinge = tf.maximum(tf.cast(0., tf.float64), dist_block)
        cap_loss_v.append(tf.reduce_sum(hinge))
    return tf.convert_to_tensor(cap_loss_v)


def distance(z1, z2):
    ## l2diff
    #return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(z1, z2)), reduction_indices=1))
    return tf.sqrt(tf.reduce_sum(tf.square(z1 - z2), 1))


if __name__ == '__main__':

    is_external_input = False
    is_case_sensitive = False
    if is_external_input:
        n_hidden_units = int(sys.argv[1])  # 10 #20 #50
        n_classes = int(sys.argv[2])  # 2
        batch_size = int(sys.argv[3])  # 100  # 450
        n_steps = int(sys.argv[4])  # 20 #200 #200
        lr = float(sys.argv[5])  # 0.001  # learning rate
        training_iters = int(sys.argv[6])  # 500  # train step
        stepwidth = int(sys.argv[7])  # 10 #10 #100
        stepwidth_testing = int(sys.argv[8])  # 1 #10 #100
        margin = int(sys.argv[9])  #  10

        # num of stacked lstm layers
        num_stacked_layers = int(sys.argv[10])  # 1

        instance_length = int(sys.argv[11])  # 3 #5  #50
        instance_step_width = int(sys.argv[12])  # 1 # 10
        L = int(sys.argv[13])  # 5 #20

        in_keep_prob = float(sys.argv[14])  # 0.5
        out_keep_prob = float(sys.argv[15])  # 1
        lambda_l2_reg = float(sys.argv[16])  # 0.001

        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[17]  # "3"
        out_prefix = sys.argv[18]  # "3"
    else:
        n_hidden_units = 2
        n_classes = 2
        batch_size = 50
        n_steps = 20
        lr = 0.001
        training_iters = 500
        stepwidth = 10
        stepwidth_testing = 1
        margin = 10

        # num of stacked lstm layers
        num_stacked_layers = 1

        instance_length = 3
        instance_step_width = 1
        L = 2

        in_keep_prob = 0.5
        out_keep_prob = 1
        lambda_l2_reg = 0.001

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        out_prefix = 'closs_reg'  # "3"

    # data_set_file = sys.argv[18]

    root_dir = '/home/ds/mnatsumeda/retail/EAP/out_test'
    data_set_dict = {}
    for i in range(3):
        filename = r'{0}/out{1:04d}.csv'.format(root_dir, i)
        training_period_mepoch = [[0, 800]]
        case_id_mepoch = [900]
        di = DataElement(filename, training_period_mepoch, case_id_mepoch)
        data_set_dict[i] = di

    if is_case_sensitive:
        test_case_dict = {}
    else:
        test_case_list = []

    for key, data_set in data_set_dict.items():
        filename = data_set.filename
        training_period_mepoch = data_set.training_period_mepoch
        case_id_mepoch = data_set.case_id_mepoch

        num_of_sample = len(["" for line in open(filename, "r")])

        n_dim = get_dimension(filename) - 1

        data = loadData(filename, num_of_sample)

        training_period = []
        for ep in training_period_mepoch:
            training_period.append(TSSUtil.getStartEndIndicies(data.ts, ep[0], ep[1]))
        print(training_period)

        tr_data = []
        tr_y = []
        for i in range(int(len(training_period))):
            start = training_period[i][0]
            end = training_period[i][1]
            label_v = [1.0, 0.0]
            tmp_tr_data, tmp_tr_y = data.generateList(n_steps, stepwidth, start, end, label_v)
            tr_data.extend(tmp_tr_data)
            tr_y.extend(tmp_tr_y)

        listNum = len(tr_data)
        print('listNum:'+str(listNum))
        if batch_size > listNum:
            batch_size = listNum
            print('set batch_size to be equal to listNum')

        case_id = []
        for ep in case_id_mepoch:
            case_id.append(TSSUtil.getStartIndex(data.ts, ep))
        print(case_id)

        aps_cap_flag = True
        batch_data_pool = batchPool(aps_cap_flag)
        label_v = [0.0, 1.0]
        for i_case in case_id:
            tr_data_case1, tr_y_case1 = data.generateRepeatList(listNum, i_case - n_steps, i_case, label_v)
            tr_data_case11, _ = data.generateRepeatList(listNum, i_case, i_case + n_steps, label_v)
            batch_data_pool.addPair_apscap(tr_data, tr_y, tr_data_case1, tr_y_case1, tr_data, [0] * listNum, tr_data_case11, [1] * listNum)

        # balancing the number of samples
        diff_num = int(len(case_id) * (len(case_id) - 1) / 2 - len(case_id))
        for i in range(diff_num):
            i_case = case_id[random.sample(len(case_id)-1, 1)]
            tr_data_case1, tr_y_case1 = data.generateRepeatList(listNum, i_case - n_steps, i_case, label_v)
            tr_data_case11, _ = data.generateRepeatList(listNum, i_case, i_case + n_steps, label_v)
            batch_data_pool.addPair_apscap(tr_data, tr_y, tr_data_case1, tr_y_case1, tr_data, [0] * listNum, tr_data_case11, [1] * listNum)
        
        # added training sample and anchor for testing
        if is_case_sensitive:
            test_case_dict[key] = []

        for i in range(len(case_id)):
            i_case = case_id[i]
            tr_data_casei, tr_y_casei = data.generateRepeatList(listNum, i_case - n_steps, i_case, label_v)
            tr_data_caseii, _ = data.generateRepeatList(listNum, i_case, i_case + n_steps, label_v)
            if is_case_sensitive:
                test_case_dict[key].append(tr_data_casei[0: batch_size])
            else:
                test_case_list.append(tr_data_casei[0: batch_size])
                # print(np.array(test_case_list).shape)
            for j in range(i, len(case_id)):
                j_case = case_id[j]
                tr_data_casej, tr_y_casej = data.generateRepeatList(listNum, j_case - n_steps, j_case, label_v)
                tr_data_casejj, _ = data.generateRepeatList(listNum, j_case, j_case + n_steps, label_v)
                batch_data_pool.addPair_apscap(tr_data_casei, tr_y_casei, tr_data_casej, tr_y_casej, tr_data_caseii, [1] * listNum, tr_data_casejj, [1] * listNum)

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
    x1 = tf.placeholder(tf.float64, [batch_size, n_steps, n_dim])
    x2 = tf.placeholder(tf.float64, [batch_size, n_steps, n_dim])
    y = tf.placeholder(tf.float64, [batch_size,])

    y1 = tf.placeholder(tf.float64, [batch_size, ])
    y2 = tf.placeholder(tf.float64, [batch_size, ])

    # x placeholder
    x11 = tf.placeholder(tf.float64, [batch_size, n_steps, n_dim])
    x22 = tf.placeholder(tf.float64, [batch_size, n_steps, n_dim])


    z1, h_list_tensor1, K1 = mi_lstm(lstm_cell, x1, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    z11, _, _ = mi_lstm(lstm_cell, x11, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    z2, h_list_tensor2, K2 = mi_lstm(lstm_cell, x2, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
    z22, _, _ = mi_lstm(lstm_cell, x22, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)

    dist_z1_z2 = distance(z1, z2)
    hingeloss = tf.maximum(tf.cast(0., tf.float64), margin - dist_z1_z2)
    clloss = tf.reduce_mean((1 -y) * tf.pow(dist_z1_z2, 2) + y * tf.pow(hingeloss, 2))

    apsloss = tf.reduce_mean(y1* aps_loss(z1, z11) + y2*aps_loss(z2, z22))

    caploss = tf.reduce_mean(y1* cap_loss(h_list_tensor1, K1)) + tf.reduce_mean(y2* cap_loss(h_list_tensor2, K2))

    # L2 regularization for weights and biases
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
            reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))


    # cost = clloss
    cost = clloss #+ 0.01*reg_loss
    # cost = clloss + apsloss
    # cost = clloss + apsloss + 0.01*reg_loss
    # cost = clloss + apsloss + caploss + 0.01*reg_loss



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
            cont_loss = 0
            aps = 0
            cap = 0
            reg = 0
            stepNum = (int)(batch_data_pool.getTrainNum()/batch_size)
            step = 0
           # batch_data_pool.reshuffle()
            while step < stepNum:
                bt1, bt2, y_label, bt11, bt22, tag1, tag2 = batch_data_pool.next_batch(batch_size)

                _, loss, contrast_loss, capLS, apsLS, regloss= sess.run([train_op, cost, clloss, caploss, apsloss, \
                                                                          reg_loss], feed_dict={x1: bt1, \
                                                                                 x2: bt2, y: y_label, x11: bt11, \
                                                                                 x22: bt22, y1: tag1, y2: tag2})
                training_loss += loss
                cont_loss += contrast_loss
                aps += apsLS
                cap += capLS
                reg += regloss
                step += 1


            training_loss_avg = training_loss / ((step + 1) * batch_size)
            cont_loss = cont_loss / ((step + 1) * batch_size)
            aps = aps / ((step + 1) * batch_size)
            cap = cap / ((step + 1) * batch_size)
            print("loss:", training_loss_avg, " contrastive loss: ", cont_loss, " cap loss:", cap, " aps loss:", aps, \
                  "reg loss:", regloss)
            epoch += 1

        for key, data_set in data_set_dict.items():
            filename = data_set.filename
            training_period_mepoch = data_set.training_period_mepoch
            case_id_mepoch = data_set.case_id_mepoch

            num_of_sample = len(["" for line in open(filename, "r")])

            n_dim = get_dimension(filename) - 1

            data = loadData(filename, num_of_sample)
            time_stamp = np.array(data.ts)
            time_stamp = time_stamp[list(range(n_steps - 1, len(data.ts), stepwidth_testing))]
            testing = data.generateTest(n_steps, batch_size, stepwidth_testing)
            if is_case_sensitive:
                test_case_list = test_case_dict[key]

            for i in range(len(case_id_mepoch)):
                test_case = test_case_list[i]
                predict = []
                dist_z1_z2_test = distance(z1, z2)
                for bt in testing:
                    # print(np.array(bt).shape, np.array(test_case).shape)
                    dist_test = dist_z1_z2_test.eval(feed_dict={x1: bt, x2: test_case})
                    predict = np.concatenate((predict, dist_test), axis=0)
                out = np.c_[time_stamp[0:predict.shape[0]], predict, 1/predict]
                out_df = pd.DataFrame(out, columns=['Time', 'distance', 'similarity'])
                out_df.to_csv(out_prefix+'_'+str(key)+'-'+str(i)+'.csv', index=False)
                # np.savetxt(str(key)+'-'+str(i)+'.csv', predict, delimiter=',')
                # print(predict.tolist())
