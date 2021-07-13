import numpy as np
import random
import tensorflow as tf

from tensorflow.contrib import rnn


class PairGenerator:
    def merge_list(self, list1, list2):
        return list1 + list2

    def __init__(self, aps_cap_flag):
        self.first = []
        self.second = []
        self.first_label = []
        self.second_label = []
        self.pair_label = []
        self.currentInd = 0

        if(aps_cap_flag):
            self.first2 = []
            self.second2 = []

            self.first_tag = []
            self.second_tag = []

            # indicate the tag for the bag containing two cases of abnormal
            self.list_tag = []


    def addPair(self, first, first_label, second, second_label):
        self.first = self.merge_list(self.first, first)
        self.first_label = self.merge_list(self.first_label, first_label)
        self.second = self.merge_list(self.second, second)
        self.second_label = self.merge_list(self.second_label, second_label)
        i = 0
        self.pair_label = []
        for bt in self.first_label:
            if np.array_equal(self.second_label[i], bt):
                self.pair_label.append(0)  
            else:
                self.pair_label.append(1)
            i += 1

    def addPair_apscap(self, first, first_label, second, second_label, first2, first_tag, second2, second_tag, list_tag):
        self.first = self.merge_list(self.first, first)
        self.first_label = self.merge_list(self.first_label, first_label)
        self.second = self.merge_list(self.second, second)
        self.second_label = self.merge_list(self.second_label, second_label)

        self.first2 = self.merge_list(self.first2, first2)
        self.second2 = self.merge_list(self.second2, second2)

        self.first_tag = self.merge_list(self.first_tag, first_tag)
        self.second_tag = self.merge_list(self.second_tag, second_tag)

        self.list_tag = self.merge_list(self.list_tag, list_tag)

        i = 0
        self.pair_label = []
        for bt in self.first_label:
            if np.array_equal(self.second_label[i], bt):
                self.pair_label.append(0)
            else:
                self.pair_label.append(1)
            i += 1

    def reshuffle(self):
        idx = np.random.permutation(len(self.first))
        self.first = np.array(self.first)[idx]
        self.first_label = np.array(self.first_label)[idx]

        self.second = np.array(self.second)[idx]
        self.second_label = np.array(self.second_label)[idx]

        if 'first2' in self.__dict__ :
            self.first2 = np.array(self.first2)[idx]
            self.first_tag = np.array(self.first_tag)[idx]
            self.second2 = np.array(self.second2)[idx]
            self.second_tag = np.array(self.second_tag)[idx]

        i = 0
        self.pair_label = []
        for bt in self.first_label:
            if np.array_equal(self.second_label[i], bt):
                self.pair_label.append(0)
            else:
                self.pair_label.append(1)
            i += 1


    def next_batch(self, batch_size):
        if self.currentInd + batch_size > len(self.first):
        #    self.reshuffle()
            if 'first2' in self.__dict__:
                bt1 = self.first[self.currentInd:]
                bt2 = self.second[self.currentInd:]
                y = self.pair_label[self.currentInd:]

                bt11 = self.first2[self.currentInd:]
                bt22 = self.second2[self.currentInd:]

                tag1 = self.first_tag[self.currentInd:]
                tag2 = self.second_tag[self.currentInd:]

                ##
                l_tag = self.list_tag[self.currentInd:]

                self.currentInd = len(self.first)

                return bt1, bt2, y, bt11, bt22, tag1, tag2, l_tag

            else:
                bt1 = self.first[self.currentInd:]
                bt2 = self.second[self.currentInd:]
                y = self.pair_label[self.currentInd:]
                self.currentInd = len(self.first)

                return bt1, bt2, y


        if 'first2' in self.__dict__:

            bt1 = self.first[self.currentInd:self.currentInd + batch_size]
            bt2 = self.second[self.currentInd:self.currentInd + batch_size]
            y = self.pair_label[self.currentInd:self.currentInd + batch_size]


            bt11 = self.first2[self.currentInd:self.currentInd + batch_size]
            bt22 = self.second2[self.currentInd:self.currentInd + batch_size]

            tag1 = self.first_tag[self.currentInd:self.currentInd + batch_size]
            tag2 = self.second_tag[self.currentInd:self.currentInd + batch_size]

            l_tag = self.list_tag[self.currentInd:self.currentInd + batch_size]

            self.currentInd += batch_size

            return bt1, bt2, y, bt11, bt22, tag1, tag2, l_tag
        else:
            bt1 = self.first[self.currentInd:self.currentInd + batch_size]
            bt2 = self.second[self.currentInd:self.currentInd + batch_size]
            y = self.pair_label[self.currentInd:self.currentInd + batch_size]
            self.currentInd += batch_size

            return bt1, bt2, y

    def hasNext(self):
        if self.currentInd  < len(self.first):
            return True
        else:
            self.currentInd = 0
            return False

    def getTrainNum(self):
        return len(self.first)