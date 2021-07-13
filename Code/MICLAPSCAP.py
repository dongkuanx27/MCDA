# encoding: utf-8
#import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER" ] ="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES" ] = "0"
import tensorflow as tf

#from tensorflow.python.ops import gen_manip_ops as _gen_manip_ops


from LoadData import LoadData as loadData
from PairGenerator import PairGenerator as batchPool

import numpy as np

def get_dimension(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        words = first_line.split(",")
        return len(words)

def merge_list(list1, list2):
    return list1 + list2

def attention_series(a_list, h_list_tensor, K, n_dim):
    # a_list: 400*1*16
    # h_list_tensor: 6400*168
    # K = 16
    # n_dim = 42

    # L1 = 2
    # w1: shape=[n_dim, 1, L1] (42*1*2)
    # V1: shape=[n_dim, L1, M/n_dim] (42*2*4)
    # U1: shape=[n_dim, L1, M/n_dim] (42*2*4)

    dim = h_list_tensor.get_shape()[1].value # dim = 168
    #d   = tf.cast(dim/n_dim, tf.int32)
    h_block_list1 = tf.reshape(h_list_tensor, [-1, n_dim, d]) # shape = 6400*42*4
    h_list_tensor1_t = tf.transpose(h_block_list1, perm=[1, 2, 0]) # shape = 42*4*6400

    temp1  = tf.matmul(V1, h_list_tensor1_t) # 42*2*6400
    temp2  = tf.matmul(U1, h_list_tensor1_t) # 42*2*6400
    temp11 = tf.tanh(temp1)
    temp22 = tf.sigmoid(temp2)
    temp3  = tf.multiply(temp11, temp22)  # 42*2*6400

    A0 = tf.matmul(w1, temp3)  # 42*1*6400
    if (len(A0.get_shape()) > 2):
        A0 = tf.squeeze(A0, squeeze_dims=1) # 42*6400
    
    A1 = tf.transpose(tf.nn.softmax(A0, dim=0)) # 6400*42

    # aa_list: 400*16
    bb_list = tf.reshape(A1, [-1, K, n_dim]) # 400*16*42
    # A1 = tf.expand_dims(A1, 2) # 6400*42*1
    # A1 = tf.tile(A1, [1,1,d]) # 6400*42*4

    h_list_tensor1_tt = tf.transpose(h_block_list1, perm=[2, 0, 1]) # 4*6400*42
    z_bt_temp = tf.multiply(h_list_tensor1_tt, A1) # 4*6400*42
    z_bt_temp = tf.transpose(z_bt_temp, perm=[1, 2, 0]) # shape = 6400*42*4
    z_bt_temp = tf.reshape(z_bt_temp, [-1, K, dim]) # 400*16*168

    # a_list: 400*1*16
    z_bt2 = tf.matmul(a_list, z_bt_temp) # 400*1*168
    if (len(z_bt2.get_shape()) > 2):
        z_bt2 = tf.squeeze(z_bt2, squeeze_dims=1) # 400*168

    return z_bt2, bb_list # z_bt2: 400*168; bb_list: 400*16*42

def attention_series_regression(a_list, h_list_tensor, K, n_dim):
    # a_list: 400*1*16
    # h_list_tensor: 6400*168
    # K = 16
    # n_dim = 42

    # L1 = 2
    # w1: shape=[n_dim, 1, L1] (42*1*2)
    # V1: shape=[n_dim, L1, M/n_dim] (42*2*4)
    # U1: shape=[n_dim, L1, M/n_dim] (42*2*4)

    # Wr: shape=[n_dim, 1, M/n_dim] (42*1*4)

    dim = h_list_tensor.get_shape()[1].value # dim = 168
    h_block_list1 = tf.reshape(h_list_tensor, [-1, n_dim, tf.cast(dim/n_dim, tf.int32)]) # shape = 6400*42*4
    h_list_tensor1_t = tf.transpose(h_block_list1, perm=[1, 2, 0]) # shape = 42*4*6400

    '''
    temp1  = tf.matmul(V1, h_list_tensor1_t) # 42*2*6400
    temp2  = tf.matmul(U1, h_list_tensor1_t) # 42*2*6400
    temp11 = tf.tanh(temp1)
    temp22 = tf.sigmoid(temp2)
    temp3  = tf.multiply(temp11, temp22)  # 42*2*6400

    A0 = tf.matmul(w1, temp3)  # 42*1*6400
    '''

    A0 = tf.matmul(Wr, h_list_tensor1_t)  # (42*1*4) * (42*4*6400) = 42*1*6400
    if (len(A0.get_shape()) > 2):
        A0 = tf.squeeze(A0, squeeze_dims=1) # 42*6400
    
    A1 = tf.transpose(tf.nn.softmax(A0, dim=0)) # 6400*42

    # aa_list: 400*16
    bb_list = tf.reshape(A1, [-1, K, n_dim]) # 400*16*42
    # A1 = tf.expand_dims(A1, 2) # 6400*42*1
    # A1 = tf.tile(A1, [1,1,d]) # 6400*42*4

    h_list_tensor1_tt = tf.transpose(h_block_list1, perm=[2, 0, 1]) # 4*6400*42
    z_bt_temp = tf.multiply(h_list_tensor1_tt, A1) # 4*6400*42
    z_bt_temp = tf.transpose(z_bt_temp, perm=[1, 2, 0]) # shape = 6400*42*4
    z_bt_temp = tf.reshape(z_bt_temp, [-1, K, dim]) # 400*16*168

    # a_list: 400*1*16
    z_bt2 = tf.matmul(a_list, z_bt_temp) # 400*1*168
    if (len(z_bt2.get_shape()) > 2):
        z_bt2 = tf.squeeze(z_bt2, squeeze_dims=1) # 400*168

    return z_bt2, bb_list # z_bt2: 400*168; bb_list: 400*16*42

def attention(h_list_tensor, K, n_dim):
    # h_list_tensor: 6400*168
    # K = 16
    # n_dim = 42

    # A = tf.transpose(tf.matmul(tf.transpose(w), tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor)))))

    A = tf.transpose(tf.nn.softmax(tf.matmul(tf.transpose(w), tf.multiply(tf.tanh(tf.matmul(V, tf.transpose(h_list_tensor))), \
                         tf.sigmoid(tf.matmul(U, tf.transpose(h_list_tensor)))))))

    dim = h_list_tensor.get_shape()[1].value # dim = 168


    h_block_list = tf.reshape(h_list_tensor, [-1, K, dim]) # 400*16*168


    A_list = tf.reshape(A, [-1, K]) # 400*16

    #z_bt1 = tf.reduce_max(h_block_list, axis=1)
    #z_bt1 = tf.reduce_mean(h_block_list, axis=1)


    a_list  = tf.nn.softmax(A_list, dim=1)
    aa_list = a_list # aa_list: 400*16
    a_list  = tf.expand_dims(a_list, 1) # a_list: 400*1*16


    z_bt1 = tf.matmul(a_list, h_block_list) # z_bt1: 400*1*168

    if (len(z_bt1.get_shape()) > 2):
        z_bt1 = tf.squeeze(z_bt1, squeeze_dims=1) # z_bt1: 400*168

    z_bt2, bb_list = attention_series(a_list, h_list_tensor, K, n_dim) # z_bt1/2: 400*168; aa_list: 400*16; bb_list: 400*16*42

    return z_bt1, aa_list, h_block_list, z_bt2, bb_list

def normalize(v):
    return tf.nn.l2_normalize(v, axis=1, epsilon=1e-12)

def mi_lstm(lstm_cell, X, instance_length, stepwidth, n_hidden_units, n_dim, num_stacked_layers):
    K = 0
    ins_list = []
    data_length = X.get_shape()[1]
    for j in range(0, data_length - instance_length + 1, stepwidth):
        instance = X[:, j:j + instance_length, :]
        ins_list.append(instance)
        K += 1

    ins_list = tf.transpose(ins_list, [1, 0, 2, 3]) #shape = 400*16*50*42
    # bt_ins = tf.concat(ins_list, axis=0)
    bt_ins = tf.reshape(ins_list, [-1, instance_length, n_dim]) # bt_ins: 6400*50*42

    # init_state = lstm_cell.zero_state(bt_ins.get_shape()[0].value, dtype=tf.float64)

    #h_list_tensor = LSTM(bt_ins, lstm_cell, n_dim, instance_length, n_hidden_units, num_stacked_layers) # h_list_tensor: 6400*168
    h_list_tensor = LSTM_variable(bt_ins, lstm_cell, n_dim, instance_length, n_hidden_units, num_stacked_layers) # h_list_tensor: 6400*168
    
    res1, atten_list1, h_b_list, res2, atten_list2 = attention(h_list_tensor, K, n_dim) # atten_list1: 400*16; h_b_list: 400*16*168; K: 16; n_dim: 42
    
    # res1, res2: 400*168
    # h_list_tensor: 6400*168
    # h_b_list: 400*16*168
    # atten_list1: 400*16
    # atten_list2: 400*16*42
    # ins_list: 400*16*50*42
    # K = 16
    return res1, h_list_tensor, K, atten_list1, h_b_list, ins_list, res2, atten_list2


def LSTM(X, cell, n_dim, n_steps, n_hidden_units, num_stacked_layers): # X: 6400*50*42
    # # X ==> (128 batches * 28 steps, 28 inputs)
    # X = tf.reshape(X, [-1, n_dim])
    #
    # # X_in = W*X + b
    # X_in = tf.matmul(X, weights['in']) + biases['in']
    # # X_in ==> (128 batches, 28 steps, 128 hidden)
    # X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    #
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, time_major=False)


    feature = final_state[num_stacked_layers - 1][1]


    return feature

def LSTM_variable(X, cell, n_dim, n_steps, n_hidden_units, num_stacked_layers):
	# X: 6400*50*42 (the dimension of '6400' is dynamic)
	# n_dim = 42
	# n_hidden_units = 168

	# Wh: shape=[n_dim,d,d] # 42*4*4
	# Wx: shape=[n_dim,d] # 42*4
	# bj: shape=[n_dim,d] # 42*4
	# W3: shape=[3*(n_dim*d),n_dim*(d+1)] # (3*(42*4)) * (42+(42*4))
	# b3: shape=[3*(n_dim*d)] # 3*(42*4)

	d = tf.cast(n_hidden_units/n_dim, tf.int32) # d = 4
	T = X.get_shape()[1].value # T = 50

	for i in range(T):
		print("index i in LSTM:", i)
		if i == 0:
			# Eq.(1)
			Wx_temp  = tf.expand_dims(Wx, 2) # 42*4*1
			xt_temp  = X[:,i,:] # 6400*42
			xt       = tf.expand_dims(tf.transpose(xt_temp), 1) # 42*1*6400
			J_t_temp = tf.matmul(Wx_temp, xt) # 42*4*6400
			J_t      = tf.tanh(tf.transpose(J_t_temp, perm=[2, 0, 1]) + bj) # 6400*42*4 + 42*4 = 6400*42*4
			vec_J_t  = tf.reshape(J_t, [-1, n_hidden_units]) # 6400*168
			
			# Eq.(3)
			c_c      = vec_J_t # 6400*168
			
			# Eq.(4)
			H_c_temp = tf.reshape(tf.tanh(c_c), [-1, n_dim, d]) # 6400*42*4
			H_c      = tf.transpose(H_c_temp, perm=[1, 2, 0]) # 42*4*6400
			
			# update H and c
			H_p = H_c # 42*4*6400
			c_p = c_c # 6400*168
		
		else:
			# Eq.(1)
			Wx_temp  = tf.expand_dims(Wx, 2) # 42*4*1
			xt_temp  = X[:,i,:] # 6400*42
			xt       = tf.expand_dims(tf.transpose(xt_temp), 1) # 42*1*6400
			J_t_temp = tf.matmul(Wh, H_p) + tf.matmul(Wx_temp, xt) # 42*4*6400 + 42*4*6400
			J_t      = tf.tanh(tf.transpose(J_t_temp, perm=[2, 0, 1]) + bj) # 6400*42*4 + 42*4 = 6400*42*4
			vec_J_t  = tf.reshape(J_t, [-1, n_hidden_units]) # 6400*168
			
			# Eq.(2)
			xt_tt    = tf.transpose(xt_temp) # 42*6400

			H_p_tt1  = tf.transpose(H_p, perm=[2, 0, 1]) # 6400*42*4
			H_p_tt2  = tf.reshape(H_p_tt1, [-1, n_hidden_units]) # 6400*168
			H_p_temp = tf.transpose(H_p_tt2) # 168*6400
			
			ifo_temp = tf.matmul(W3, tf.concat([xt_tt, H_p_temp], 0)) # ((3*168)*(42+42*4)) * ((42+168)*6400) = (3*168)*6400
			ifo      = tf.sigmoid(tf.transpose(ifo_temp) + b3) # 6400*(3*168)
			
			# Eq.(3)
			it       = ifo[:,0:n_hidden_units] # 6400*168
			ft       = ifo[:,n_hidden_units:n_hidden_units*2] # 6400*168
			ot       = ifo[:,n_hidden_units*2:n_hidden_units*3] # 6400*168
			c_c      = tf.multiply(ft, c_p) + tf.multiply(it, vec_J_t) # 6400*168
			
			# Eq.(4)
			H_c_temp = tf.multiply(ot, tf.tanh(c_c)) # 6400*168
			H_c_tt   = tf.reshape(H_c_temp, [-1, n_dim, d]) # 6400*42*4
			H_c      = tf.transpose(H_c_tt, perm=[1, 2, 0]) # 42*4*6400

			# update H and c
			H_p = H_c # 42*4*6400
			c_p = c_c # 6400*168

	h_list_tensor_temp = tf.transpose(H_p, perm=[2, 0, 1])  # 6400*42*4
	h_list_tensor      = tf.reshape(h_list_tensor_temp, [-1, n_hidden_units]) # h_list_tensor: 6400*168

	return h_list_tensor

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


def pos_ins_loss(lab1, lab2, a1, a2, h1, h2, lab3):
    # lab1, lab2: "first_tag" and "second_tag"; 400; tensor "list"
    # a1, a2: attention; 400*16
    # h1, h2: 400*16*50
    # l_t: 400; tensor "list"
    
    '''
    idx1 = tf.where(tf.equal(lab1, tf.constant(1.0, tf.float64)))
    idx2 = tf.where(tf.equal(lab2, tf.constant(1.0, tf.float64)))
    '''
    idx12 = tf.where(tf.equal(lab3, tf.constant(1.0, tf.float64)))

    sub_a1 = tf.gather(a1, tf.transpose(idx12))[0]
    sub_a2 = tf.gather(a2, tf.transpose(idx12))[0]

    idx11 = tf.argmax(sub_a1, 1)
    idx22 = tf.argmax(sub_a2, 1)

    # idx matrix: shape = [idx12, [idx11']]
    idxm1 = tf.concat([idx12, tf.expand_dims(idx11,1)], 1)
    idxm2 = tf.concat([idx12, tf.expand_dims(idx22,1)], 1)

    sub_h1 = tf.gather_nd(h1, idxm1) # 2-dimension
    sub_h2 = tf.gather_nd(h2, idxm2)
    sub_h  = tf.concat([sub_h1, sub_h2], 0)

    # for case of two abnormal with max operation
    posinsloss = tf.losses.cosine_distance(tf.cast(tf.nn.l2_normalize(sub_h1, 0), tf.float32), tf.cast(tf.nn.l2_normalize(sub_h2, 0), tf.float32), dim=0) # cosine_sim = 1 - posinsloss
    posinsloss = posinsloss * tf.cast(tf.reduce_max(lab3), tf.float32)
    '''
    sub_h_norm = tf.nn.l2_normalize(sub_h, dim=1)
    dist_mat = 1 - tf.matmul(sub_h_norm, sub_h_norm, transpose_b=True) # transpose second matrix
    dist_mat_1 = tf.cast(dist_mat > 1e-6, dist_mat.dtype) * dist_mat
    posinsloss = tf.reduce_sum(dist_mat_1) # ! This is an approximation
    '''

    #posinsloss = 1
    
    return tf.cast(posinsloss, tf.float64)



if __name__ == '__main__':

	filename = 'xxxxxx'

	num_of_sample = len(["" for line in open(filename, "r")]) # num_of_sample = 814536

	n_dim = get_dimension(filename) - 1 # n_dim = 6
	print("n_dim:", n_dim)
	data = loadData(filename, num_of_sample)

	d = 15 #tf.cast(n_hidden_units/n_dim, tf.int32)
	n_hidden_units = d * n_dim # 6*15

	n_classes = 2
	batch_size = 40 # 400
	n_steps = 100
	lr = 0.001  # learning rate
	training_iters = 500  # train step
	stepwidth = 50
	margin = 10 # ??? < 1.2

	# num of stacked lstm layers
	num_stacked_layers = 1

	instance_length = 25
	instance_step_width = 5
	L = 20
	
	L1 = 7


	in_keep_prob  = 1 #0.5
	out_keep_prob = 1
	lambda_l2_reg = 0.001

	model_path = "xxxxxx.ckpt"

	tf.reset_default_graph()

	weights = {'in': tf.get_variable('Weights_in', shape=[n_dim, n_hidden_units], dtype=tf.float64, initializer=tf.truncated_normal_initializer()),'out': tf.get_variable('Weights_out', shape=[n_hidden_units, n_classes], dtype=tf.float64, initializer=tf.truncated_normal_initializer()),}
	biases = {'in': tf.get_variable('Biases_in', shape=[n_hidden_units, ], dtype=tf.float64, initializer=tf.constant_initializer(0.)), 'out': tf.get_variable('Biases_out', shape=[n_classes, ], dtype=tf.float64, initializer=tf.constant_initializer(0.)),}

	# lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	cells = []
	for i in range(num_stacked_layers):
		with tf.variable_scope('RNN_{}'.format(i)):
			cell = tf.contrib.rnn.LSTMCell(n_hidden_units, use_peepholes=True)
			cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob)
			cells.append(cell_dropout)
	lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

	M = n_hidden_units
	w = tf.Variable(tf.constant(0.1, shape=[L, 1], dtype=tf.float64)) # L = 20, M = 168
	V = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))
	U = tf.Variable(tf.constant(0.1, shape=[L, M], dtype=tf.float64))

	weights = {'w1': tf.get_variable('Weights_w1', shape=[n_dim, 1, L1], dtype=tf.float64, \
                                     initializer=tf.truncated_normal_initializer()), \
               'V1': tf.get_variable('Weights_V1', shape=[n_dim, L1, M/n_dim], dtype=tf.float64, \
                                     initializer=tf.truncated_normal_initializer()), \
               'U1': tf.get_variable('Weights_U1', shape=[n_dim, L1, M/n_dim], dtype=tf.float64, \
                                     initializer=tf.truncated_normal_initializer())}

	# para for variable attention: Max_Welling
	# w1 = tf.Variable(tf.constant(0.1, shape=[n_dim, 1, L1], dtype=tf.float64)) # n_dim = 42
	# V1 = tf.Variable(tf.constant(0.1, shape=[n_dim, L1, M/n_dim], dtype=tf.float64))
	# U1 = tf.Variable(tf.constant(0.1, shape=[n_dim, L1, M/n_dim], dtype=tf.float64))


	w1, V1, U1 = weights['w1'], weights['V1'], weights['U1']


    # para for variable attention: regression
	Wr = tf.Variable(tf.constant(0.1, shape=[n_dim, 1, M/n_dim], dtype=tf.float64))

	# parameters of LSTM
	Wh = tf.Variable(tf.truncated_normal(shape=[n_dim,d,d], stddev=0.1, dtype=tf.float64)) # 42*4*4
	Wx = tf.Variable(tf.truncated_normal(shape=[n_dim,d], stddev=0.1, dtype=tf.float64)) # 42*4
	bj = tf.Variable(tf.truncated_normal(shape=[n_dim,d], stddev=0.1, dtype=tf.float64)) # 42*4
	W3 = tf.Variable(tf.truncated_normal(shape=[3*(n_dim*d),n_dim*(d+1)], stddev=0.1, dtype=tf.float64)) # (3*(42*4)) * (42+(42*4))
	b3 = tf.Variable(tf.truncated_normal(shape=[3*(n_dim*d)], stddev=0.1, dtype=tf.float64)) # 3*(42*4)

	# x placeholder
	x1 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
	x2 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
	y  = tf.placeholder(tf.float64, [None,])

	y1 = tf.placeholder(tf.float64, [None, ])
	y2 = tf.placeholder(tf.float64, [None, ])

	# x placeholder
	x11 = tf.placeholder(tf.float64, [None, n_steps, n_dim])
	x22 = tf.placeholder(tf.float64, [None, n_steps, n_dim])

	# indicating whether the bag should be utilized to find positive instance
	list_tag = tf.placeholder(tf.float64, [None, ])

	start   = 0 # 1
	end     = 1000 # 100000
	case_id = [1200, 1400]  # 513081
	label_v = [1.0, 0.0]
	tr_data, tr_y = data.generateList(n_steps, stepwidth, start, end, label_v)

	listNum = len(tr_data)
	label_v = [0.0, 1.0]
	tr_data_case1, tr_y_case1 = data.generateRepeatList(listNum, case_id[0] - n_steps, case_id[0], label_v)
	tr_data_case11, _ = data.generateRepeatList(listNum, case_id[0], case_id[0]+n_steps, label_v)

	# list of tag for the 2nd "999" and the 4th "999"
	list_tag1 = [1] + [0] * (listNum - 1)
	bgn_tag, end_tag = (listNum * 1) // batch_size, (listNum * 2) // batch_size
	for i in range(end_tag - bgn_tag):
		idx = (bgn_tag + (i + 1)) * batch_size - listNum * 1
		list_tag1[idx] = 1

	list_tag2 = [1] + [0] * (listNum - 1)
	bgn_tag, end_tag = (listNum * 3) // batch_size, (listNum * 4) // batch_size
	for i in range(end_tag - bgn_tag):
		idx = (bgn_tag + (i + 1)) * batch_size - listNum * 3
		list_tag2[idx] = 1

	aps_cap_flag = True
	batch_data_pool = batchPool(aps_cap_flag)
	batch_data_pool.addPair_apscap(tr_data, tr_y, tr_data_case1, tr_y_case1, tr_data, [0] * listNum, tr_data_case11, [1] * listNum, [0] * listNum)
	label_v = [0.0, 1.0]
	tr_data_case2, tr_y_case2 = data.generateRepeatList(listNum, case_id[1] - n_steps, case_id[1], label_v)
	tr_data_case22, _ = data.generateRepeatList(listNum, case_id[1], case_id[1] + n_steps, label_v)

	batch_data_pool.addPair_apscap(tr_data_case1, tr_y_case1, tr_data_case2, tr_y_case2, tr_data_case11, [1] * listNum, tr_data_case22, [1] * listNum, list_tag1)

	batch_data_pool.addPair_apscap(tr_data, tr_y, tr_data_case2, tr_y_case2, tr_data, [0] * listNum, tr_data_case22, [1] * listNum, [0] * listNum)

	batch_data_pool.addPair_apscap(tr_data_case2, tr_y_case2, tr_data_case1, tr_y_case1, tr_data_case22, [1] * listNum, tr_data_case11, [1] * listNum, list_tag2)


    # a_list1: 400*16; h_list1: 400*16*168
    # ins_list1: 400*16*50*42

    ## res1, res2: 400*168
    ## h_list_tensor: 6400*168
    ## h_b_list: 400*16*168
    ## atten_list1: 400*16
    ## atten_list2: 400*16*42
    ## ins_list: 400*16*50*42
    ## K = 16
    # return: res1, h_list_tensor, K, atten_list1, h_b_list, ins_list, res2, atten_list2

	_, h_list_tensor1, K1, a_list1, h_list1, ins_list1, z1, a_list11 = mi_lstm(lstm_cell, x1, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
	_, _, _, _, _, _, z11, _ = mi_lstm(lstm_cell, x11, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
	_, h_list_tensor2, K2, a_list2, h_list2, ins_list2, z2, a_list22 = mi_lstm(lstm_cell, x2, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)
	_, _, _, _, _, _, z22, _ = mi_lstm(lstm_cell, x22, instance_length, instance_step_width, n_hidden_units, n_dim, num_stacked_layers)


	z1 = normalize(z1)
	z2 = normalize(z2)

	dist_z1_z2 = distance(z1, z2)
	hingeloss = tf.maximum(tf.cast(0., tf.float64), margin - dist_z1_z2)
	clloss = tf.reduce_mean((1 -y) * tf.pow(dist_z1_z2, 2) + y * tf.pow(hingeloss, 2))

	apsloss = tf.reduce_mean(y1* aps_loss(z1, z11) + y2*aps_loss(z2, z22))

	caploss = tf.reduce_mean(y1* cap_loss(h_list_tensor1, K1)) + tf.reduce_mean(y2* cap_loss(h_list_tensor2, K2))

	# positive-instance loss for emphasizing on the attention of positive instances
	# pos_ins_loss = 0
	posloss = pos_ins_loss(y1, y2, a_list1, a_list2, h_list1, h_list2, list_tag)


	# L2 regularization for weights and biases
	reg_loss = 0
	for tf_var in tf.trainable_variables():
		if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
			reg_loss += lambda_l2_reg * tf.reduce_mean(tf.nn.l2_loss(tf_var))


	# cost = clloss + apsloss  + 0.01*reg_loss #+ caploss
	# cost = clloss #+ 200*posloss
	cost = clloss + 0.0005*reg_loss

	train_op = tf.train.AdamOptimizer(lr).minimize(cost)

	# optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	# gvs = optimizer.compute_gradients(cost)
	# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	# train_op = optimizer.apply_gradients(capped_gvs)

	init = tf.global_variables_initializer()

	
	saver = tf.train.Saver()

	# temp_list = []
	# for i in range(0, batch_size):
	#     ins = h_list_tensor1[i * K1:(i + 1) * K1]
	#     temp = distance2(z2[i], ins)
	#     temp_list = merge_list(temp_list, temp)
	# dist_z1_z2_test = tf.convert_to_tensor(temp_list)

	dist_z1_z2_test = distance(z1, z2)

	with tf.Session() as sess:
		sess.run(init)
		epoch = 0

		List_training_loss_avg = []
		List_cont_loss         = []
		List_cap               = []
		List_aps               = []
		List_pos               = []

		while epoch < training_iters:

			print("epoch:", epoch)

			training_loss = 0
			cont_loss = 0
			aps = 0
			cap = 0
			reg = 0

			pos = 0

			stepNum = (int)(batch_data_pool.getTrainNum()/batch_size)
			step = 0
			# batch_data_pool.reshuffle()
			while batch_data_pool.hasNext():
				bt1, bt2, y_label, bt11, bt22, tag1, tag2, l_tag = batch_data_pool.next_batch(batch_size)

				# ins_list: 400*16*50*42
				# h_list: 400*16*168
				_, loss, contrast_loss, capLS, apsLS, regloss, posLS, a_1, a_2, ins_1, ins_2, a_11, a_22, h1, h2  = sess.run([train_op, cost, clloss, caploss, apsloss, reg_loss, posloss, a_list1, a_list2, ins_list1, ins_list2, a_list11, a_list22, h_list1, h_list2], feed_dict={x1:bt1, x2:bt2, y:y_label, x11:bt11, x22:bt22, y1:tag1, y2:tag2, list_tag:l_tag})

				training_loss += loss
				cont_loss += contrast_loss
				aps += apsLS
				cap += capLS
				reg += regloss

				pos += posLS
				print("poss loss at batch-level:", posLS)
				#print("hidden representation of h1:", h1)


				# store attention and original data of the positive bag after well trained
				if epoch == (training_iters-1) and step == (listNum*4 // batch_size):
				#if epoch == (0) and step == (listNum*4 // batch_size):
					#idx_pos = np.asarray(np.where(np.asarray(l_tag) == 0)[0])
					idx_pos = l_tag.index(1)

					print("l_tag:", l_tag)
					print("idx_pos:", idx_pos)
					print("idx_pos type:", type(idx_pos))

					data_ins1, data_ins2   = ins_1[idx_pos,:,:,:], ins_2[idx_pos,:,:,:] # ins_1 is array: 400*16*50*42
					data_bag1, data_bag2   = bt1[idx_pos], bt2[idx_pos] # bt1 is list: 400 * (200*42); the original data of a bag
					data_att1, data_att2   = a_1[idx_pos,:], a_2[idx_pos,:] # a_1 is array: 400*16; 16 instances
					data_att11, data_att22 = a_11[idx_pos,:,:], a_22[idx_pos,:,:] # a_11 is tensor: 400*16*42

					np.savez("/home/dsi/dxu/Research_Server/Working/Deep_MIL/For_synthetic_data/EAP_dual_attention_Max_Welling/result_saved/ins_bag_att.npz", ins1 = data_ins1, ins2 = data_ins2, bag1 = data_bag1, bag2 = data_bag2, att1 = data_att1, att2 = data_att2, att11 = data_att11, att22 = data_att22)

				step += 1

			training_loss_avg = training_loss / (batch_data_pool.getTrainNum())
			cont_loss = cont_loss / ((step + 1) * batch_size)
			aps = aps / ((step + 1) * batch_size)
			cap = cap / ((step + 1) * batch_size)
			pos = pos / ((step + 1) * batch_size)

			List_training_loss_avg = np.append(List_training_loss_avg, training_loss_avg)
			List_cont_loss         = np.append(List_cont_loss, cont_loss)
			List_cap               = np.append(List_cap, cap)
			List_aps               = np.append(List_aps, aps)
			List_pos               = np.append(List_pos, pos)

			print("loss:", training_loss_avg, " contrastive loss: ", cont_loss, " cap loss:", cap, " aps loss:", aps, "reg loss:", regloss, "pos loss:", pos)
			print('')

			'''
			fetch = {'obj_gmm':obj_gmm}
			RR = sess.run(fetch,feed_dict={keep_prob: dropout, train_x_v: train_norm_x, train_x_v_col: train_norm_x_col})            
			print("Joint-GMM-training Epoch %d: obj_gmm %g; obj_cross %g;"
			      % (k, RR_gmm['obj_gmm'], RR_gmm['obj_cross']))
			'''

			epoch += 1




		save_path = saver.save(sess, model_path)
		print("Model saved in file: %s" % save_path)

		# save all loss
		np.savez("xxxxxx/all_loss.npz", List_training_loss_avg = List_training_loss_avg, List_cont_loss = List_cont_loss, List_cap = List_cap, List_aps = List_aps, List_pos = List_pos)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
	

	print("Starting 2nd session...")
	with tf.Session() as sess:
		
		sess.run(init)

		load_path = saver.restore(sess, model_path)
		# print("Model restored from file: %s" % save_path)

		###############################################

		filename = 'xxxxxx'

		num_of_sample = len(["" for line in open(filename, "r")])

		n_dim = get_dimension(filename) - 2

		data = loadData(filename, num_of_sample)

		data.delete(17, axis=1)

		testing = data.generateTest(n_steps, batch_size, stepwidth)

		test_case = tr_data_case2[0: batch_size]

		case_id = [36000]

		predict = []

		for bt in testing:
			dist_test = dist_z1_z2_test.eval(feed_dict={x1: bt, x2: test_case})
			predict = np.concatenate((predict, dist_test), axis=0)

		print(predict.tolist())
	

    