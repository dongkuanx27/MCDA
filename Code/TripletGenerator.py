import numpy as np
import random
import tensorflow as tf

from tensorflow.contrib import rnn


class TripletGenerator:
    def merge_list(self, list1, list2):
        return list1 + list2

    def __init__(self, aps_cap_flag):
        self.anchor = []
        self.positive = []
        self.negative = []
        self.currentInd = 0

        if(aps_cap_flag):
            self.anchor2 = []
            self.anchor_tag = []
            self.pos2 = []
            self.pos_tag = []
            self.neg2 = []
            self.neg_tag = []

    def addTriplet(self, anchor, positive, negative):

        self.anchor = self.merge_list(self.anchor, anchor)
        self.positive = self.merge_list(self.positive, positive)
        self.negative = self.merge_list(self.negative, negative)

    def addTriplet_apscap(self, anchor, positive, negative, anchor2, anchor_tag, pos2, pos_tag, neg2, neg_tag):
        self.anchor = self.merge_list(self.anchor, anchor)
        self.positive = self.merge_list(self.positive, positive)
        self.negative = self.merge_list(self.negative, negative)

        self.anchor2 = self.merge_list(self.anchor2, anchor2)
        self.anchor_tag = self.merge_list(self.anchor_tag, anchor_tag)
        self.pos2 = self.merge_list(self.pos2, pos2)
        self.pos_tag = self.merge_list(self.pos_tag, pos_tag)
        self.neg2 = self.merge_list(self.neg2, neg2)
        self.neg_tag = self.merge_list(self.neg_tag, neg_tag)

    def reshuffle(self):
        idx = np.random.permutation(len(self.anchor))
        self.positive = np.array(self.positive)[idx]
        self.anchor = np.array(self.anchor)[idx]
        self.negative = np.array(self.negative)[idx]

        if 'anchor2' in self.__dict__:
            self.anchor2 = np.array(self.anchor2)[idx]
            self.anchor_tag = np.array(self.anchor_tag)[idx]
            self.pos2 = np.array(self.pos2)[idx]
            self.pos_tag = np.array(self.pos_tag)[idx]
            self.neg2 = np.array(self.neg2)[idx]
            self.neg_tag = np.array(self.neg_tag)[idx]


    def next_batch(self, batch_size):
        if self.currentInd + batch_size > len(self.anchor):
        #    self.reshuffle()
            if 'anchor2' in self.__dict__:
                bt1 = self.anchor[self.currentInd:]
                bt2 = self.positive[self.currentInd:]
                bt3 = self.negative[self.currentInd:]

                bt11 = self.anchor2[self.currentInd:]
                bt22 = self.pos2[self.currentInd:]
                bt33 = self.neg2[self.currentInd:]
                bt_tag1 = self.anchor_tag[self.currentInd:]
                bt_tag2 = self.pos_tag[self.currentInd:]
                bt_tag3 = self.neg_tag[self.currentInd:]

                self.currentInd = len(self.anchor)

                return bt1, bt2, bt3, bt11, bt22, bt33, bt_tag1, bt_tag2, bt_tag3

            else:

                bt1 = self.anchor[self.currentInd:]
                bt2 = self.positive[self.currentInd:]
                bt3 = self.negative[self.currentInd:]
                self.currentInd = len(self.anchor)

                return bt1, bt2, bt3

        if 'anchor2' in self.__dict__:
            bt1 = self.anchor[self.currentInd:self.currentInd + batch_size]
            bt2 = self.positive[self.currentInd:self.currentInd + batch_size]
            bt3 = self.negative[self.currentInd:self.currentInd + batch_size]

            bt11 = self.anchor2[self.currentInd:self.currentInd + batch_size]
            bt22 = self.pos2[self.currentInd:self.currentInd + batch_size]
            bt33 = self.neg2[self.currentInd:self.currentInd + batch_size]
            bt_tag1 = self.anchor_tag[self.currentInd:self.currentInd + batch_size]
            bt_tag2 = self.pos_tag[self.currentInd:self.currentInd + batch_size]
            bt_tag3 = self.neg_tag[self.currentInd:self.currentInd + batch_size]

            self.currentInd += batch_size

            return bt1, bt2, bt3, bt11, bt22, bt33, bt_tag1, bt_tag2, bt_tag3

        else:

            bt1 = self.anchor[self.currentInd:self.currentInd + batch_size]
            bt2 = self.positive[self.currentInd:self.currentInd + batch_size]
            bt3 = self.negative[self.currentInd:self.currentInd + batch_size]
            self.currentInd += batch_size

            return bt1, bt2, bt3

    def getTrainNum(self):
        return len(self.anchor)
    def hasNext(self):
        if self.currentInd < len(self.anchor):
            return True
        else:
            self.currentInd = 0
            return False