import numpy as np
import random
import tensorflow as tf

from tensorflow.contrib import rnn


class LoadData2:
    @staticmethod
    def minmax_normalization(x, base):
        min_val = np.min(base, axis=0)
        max_val = np.max(base, axis=0)
        norm_x = (x - min_val) / (max_val - min_val + 1e-12)
        # print norm_x
        return norm_x

    def __init__(self, filename):
        self.num_dim = 0
        self.xy = []

        fin = open(filename, 'r')
        line = fin.readline()
        while line != '':
            line = line.strip()
            field = line.split(',')
            p = []
            for i in range(0, len(field)):  ##############remove first column time index
                p.append(float(field[i]))
            line = fin.readline()
            self.xy.append(p)
        fin.close()

        base = np.concatenate(self.xy, 0)
        self.xy = self.minmax_normalization(self.xy, base)

        self.num_points = len(self.xy)
        self.num_dim = len(self.xy[0])

        print('# dimensions: ' + str(len(self.xy[0])))

    def getData(self):
        return self.xy

    def getNumPoints(self):
        return self.num_points

    def delete(self, i, axis):
        self.xy = np.delete(self.xy, i, axis)

    def getBatch(self, list, batchsize):
        res_list = []
        size_list = len(list)
        round = (int)(size_list/batchsize)
        for i in range(round):
            res_list.append(np.array(list[i*batchsize:(i+1)*batchsize]))
        return res_list


    def generateSeqs(self, inputLength, batch_num, stepwidth, outputLength):
        # Constants
        base = np.concatenate([self.train_data, self.test_data], 0)
        self.train_data = self.minmax_normalization(self.train_data, base)
        self.test_data = self.minmax_normalization(self.test_data, base)

        elem_num = len(self.xy[0])

        lenth_tr = len(self.train_data)

        tr_batch_list = []
        tr_target_batch_list = []
        for j in range(0, lenth_tr - inputLength + 1 - outputLength, stepwidth):
            tr_batch_list.append(self.train_data[j:j + inputLength])
            tr_target_batch_list.append(np.array(self.train_data[j+inputLength:j + inputLength+outputLength]))
        tr_batch_list = self.getBatch(tr_batch_list, batch_num)
        tr_target_batch_list = self.getBatch(tr_target_batch_list, batch_num)

        te_batch_list = []
        te_target_batch_list = []
        for j in range(0, self.num_points - inputLength + 1 - outputLength, outputLength):
            te_batch_list.append(np.array(self.test_data[j:j + inputLength]))
            te_target_batch_list.append(np.array(self.test_data[j + inputLength:j + inputLength + outputLength]))
        te_batch_list = self.getBatch(te_batch_list, batch_num)
        te_target_batch_list = self.getBatch(te_target_batch_list, batch_num)

        return tr_batch_list, tr_target_batch_list, te_batch_list, te_target_batch_list

    def generateTrTe(self, inputLength, batch_num, stepwidth):
        # Constants
        base = np.concatenate([self.train_data, self.test_data], 0)
        self.train_data = self.minmax_normalization(self.train_data, base)
        self.test_data = self.minmax_normalization(self.test_data, base)


        lenth_tr = len(self.train_data)

        tr_batch_list = []
        for j in range(0, lenth_tr - inputLength + 1, stepwidth):
            tr_batch_list.append(self.train_data[j:j + inputLength])
        tr_batch_list = self.getBatch(tr_batch_list, batch_num)

        te_batch_list = []
        for j in range(0, self.num_points - inputLength + 1, 1):
            te_batch_list.append(np.array(self.test_data[j:j + inputLength]))
        te_batch_list = self.getBatch(te_batch_list, batch_num)

        return tr_batch_list, te_batch_list

    def generateBatch(self, inputLength, batch_num, stepwidth, start, end, label_v):
        data = self.xy[start:end+1]

        lenth_tr = len(data)

        batch_list = []
        batch_y = []
        for j in range(0, lenth_tr - inputLength + 1, stepwidth):
            batch_list.append(data[j:j + inputLength])
            batch_y.append(label_v)
        batch_list = self.getBatch(batch_list, batch_num)
        batch_y = self.getBatch(batch_y, batch_num)

        return batch_list, batch_y

    def generateRepeatBatch(self, batch_size, listNum, start, end, label_v):
        element = self.xy[start:end]
        res = []
        res_y = []
        for i in range(0, listNum):
            batch_list = []
            batch_list_y =[]
            for j in range(0, batch_size):
                batch_list.append(element)
                batch_list_y.append(label_v)
            res.append(np.array(batch_list))
            res_y.append(np.array(batch_list_y))
        return res, res_y

    def generateList(self, inputLength, stepwidth, start, end, label_v):
        data = self.xy[start:end+1]

        lenth_tr = len(data)

        batch_list = []
        batch_y = []
        for j in range(0, lenth_tr - inputLength + 1, stepwidth):
            batch_list.append(data[j:j + inputLength])
            batch_y.append(label_v)

        return batch_list, batch_y

    def generateRepeatList(self, listNum, start, end, label_v):
        element = self.xy[start:end]
        batch_list = []
        batch_list_y =[]
        for i in range(0, listNum):
            batch_list.append(element)
            batch_list_y.append(label_v)
        return batch_list, batch_list_y

    def generateTest(self, inputLength, batch_num, stepwidth):
        data = self.xy

        lenth_tr = len(data)

        batch_list = []
        for j in range(0, lenth_tr - inputLength + 1, stepwidth):
            batch_list.append(data[j:j + inputLength])
        batch_list = self.getBatch(batch_list, batch_num)

        return batch_list

if __name__ == '__main__':
    data = LoadData('../../../../dataset/arrhythmia/heart.pp.csv')
    # usenet.tfidf_process()
    # train_data, test_data, test_label, num_test_points, num_dirty_points = kddcup10.get_clean_training_testing_data(0.5)
    # print np.shape(train_data), np.shape(test_data), num_test_points, num_dirty_points
