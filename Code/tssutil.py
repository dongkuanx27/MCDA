import csv

class TSSUtil:
    @staticmethod
    def getStartIndex(time_indices, tr_start):
        cnt = 0
        for time_index in time_indices:
            timeEpochMS = float(time_index)
            # print(timeEpochMS)
            if timeEpochMS >= float(tr_start):
                tr_start_id = cnt
                break
            cnt = cnt + 1
        return tr_start_id

    @staticmethod
    def getStartEndIndicies(time_indices, tr_start, tr_end):
        cnt = 0
        isFindStart = True
        isFindEnd = False
        for time_index in time_indices:
            timeEpochMS = float(time_index)
            # print(timeEpochMS)
            if isFindStart:
                if timeEpochMS >= float(tr_start):
                    tr_start_id = cnt
                    isFindStart = False
                    isFindEnd = True
            elif isFindEnd:
                if timeEpochMS >= float(tr_end):
                    tr_end_id = cnt
                    isFindEnd = False
                    break
            cnt = cnt + 1
        return [tr_start_id, tr_end_id]

    @staticmethod
    def getStartEndIndiciesWithFile(filename, tr_start, tr_end):
        cnt = 0
        isFindStart = True
        isFindEnd = False
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            # header = next(csvreader)
            next(csvreader)
            for row in csvreader:
                timeEpochMS = float(row[0])
                # print(timeEpochMS)
                if isFindStart:
                    if timeEpochMS >= float(tr_start):
                        tr_start_id = cnt
                        isFindStart = False
                        isFindEnd = True
                elif isFindEnd:
                    if timeEpochMS >= float(tr_end):
                        tr_end_id = cnt
                        isFindEnd = False
                        break
                cnt = cnt + 1
        return [tr_start_id, tr_end_id]
