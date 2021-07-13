
import os
import numpy as np

from LoadData2 import LoadData2 as loadData

class mill:
    def merge_list(self, list1, list2):
        return list1 + list2

    def minmax_normalization(self, x, base):
        min_val = np.min(base, axis=0)
        max_val = np.max(base, axis=0)
        norm_x = (x - min_val) / (max_val - min_val + 1e-12)
        # print norm_x
        return norm_x


    def __init__(self, p, n_steps, stepwidth):
        res_x =[]
        res_y = []
        loadfile = []
        loadfile.append(p)
        label = 0
        while (loadfile):
            try:
                path = loadfile.pop()

                foldname = path.rsplit('/', 1)[-1]
                if foldname.isdigit() :
                    label = int(foldname)

                # print path
                for x in os.listdir(path):
                    if x.startswith('.'):
                        continue
                    if os.path.isfile(os.path.join(path, x)):
                        print(x)

                        data = loadData(os.path.join(path, x))
                        start = 0
                        end = data.getNumPoints()
                        label_v = np.zeros([16], dtype=int)
                        label_v[label-1] = 1

                    #    xx, yy = data.generateList(n_steps, stepwidth, start, end, label_v)
                    #    res_x = self.merge_list(res_x, xx)
                    #    res_y = self.merge_list(res_y, yy)
                        res_x.append(data.xy)
                        res_y.append(label_v)

                    else:
                        loadfile.append(os.path.join(path, x))
            except Exception as e:
                print(str(e) + path)
        self.x = res_x
        self.y = res_y

    def getData(self):
        return self.x, self.y

    def getBatch(self, list, batchsize):
        res_list = []
        size_list = len(list)
        round = (int)(size_list/batchsize)
        for i in range(round):
            res_list.append(np.array(list[i*batchsize:(i+1)*batchsize]))
        return res_list

    def generateBatchList(self, batch_size):

        batch_list_x = self.getBatch(self.x, batch_size)
        batch_list_y = self.getBatch(self.y, batch_size)

        return batch_list_x, batch_list_y

