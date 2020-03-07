import pandas as pd
import hashlib
import array
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
from sklearn.utils.murmurhash import murmurhash3_32
from random import choice
from queue import PriorityQueue
import heapdict

class CountSketch(object):
    # d is the number of hash functions, R is the range of hash functions
    def __init__(self, d, R):
        self.d = d
        self.R = R
        self.table = []
        for _ in range(d):
            bucket = array.array("l", (0 for _ in range(R)))
            self.table.append(bucket)

    def add(self, element):
        # TODO minsketch是直接计数+1， count sketch是加上无偏估计量g, g 映射到 {-1,1}
        for bucket, i, j in zip(self.table, self.hashfunc(element), self.hash_binary_func(element)):
            bucket[i] = bucket[i] + j

    def query(self, element):
        templist = []
        for bucket, i, j in zip(self.table, self.hashfunc(element), self.hash_binary_func(element)):
            templist.append(bucket[i] * j)
        templist.sort()
        return templist[self.d // 2]

    def hashfunc(self, element):
        templist = []
        for i in range(self.d):
            if i == 0:
                templist.append(murmurhash3_32(element, seed=2 ** 30 - 1, positive=True) % self.R)
            elif i == 1:
                templist.append(murmurhash3_32(element, seed=2 ** 32 - 1, positive=True) % self.R)
            elif i == 2:
                templist.append(murmurhash3_32(element, seed=2 ** 20 - 1, positive=True) % self.R)
            elif i == 3:
                templist.append(murmurhash3_32(element, seed=2 ** 25 - 1, positive=True) % self.R)
        return templist

    def hash_binary_func(self, element):
        templist = []
        for i in range(self.d):
            if i == 0:
                sign = murmurhash3_32(element, seed=2 ** 32 - 1, positive=True) % 2
                if sign == 0:
                    templist.append(-1)
                else:
                    templist.append(1)

            elif i == 1:
                sign = murmurhash3_32(element, seed=2 ** 30 - 1, positive=True) % 2
                if sign == 0:
                    templist.append(-1)
                else:
                    templist.append(1)
            elif i == 2:
                sign = murmurhash3_32(element, seed=2 ** 25 - 1, positive=True) % 2
                if sign == 0:
                    templist.append(-1)
                else:
                    templist.append(1)
            elif i == 3:
                sign = murmurhash3_32(element, seed=2 ** 20 - 1, positive=True) % 2
                if sign == 0:
                    templist.append(-1)
                else:
                    templist.append(1)

        return templist

    def __getitem__(self, element):
        return self.query(element)


class CountMinSketch(object):
    def __init__(self, d, R):  # d is the number of hash functions, R is the range of hash functions
        self.d = d
        self.R = R
        self.table = []
        # construct d * R table
        for _ in range(d):
            bucket = array.array("l", (0 for _ in range(R)))
            self.table.append(bucket)

    def add(self, element, incresement=1):
        for table, i in zip(self.table, self.hashfunc(element)):
            table[i] += incresement

    def query(self, element):
        temp = min(table[i] for table, i in zip(self.table, self.hashfunc(element)))
        return temp

    def __getitem__(self, element):
        return self.query(element)

    # the hash functions
    def hashfunc(self, element):
        templist = []
        for i in range(self.d):
            if i == 0:
                templist.append(murmurhash3_32(element, seed=2 ** 30 - 1, positive=True) % self.R)
            elif i == 1:
                templist.append(murmurhash3_32(element, seed=2 ** 32 - 1, positive=True) % self.R)
            elif i == 2:
                templist.append(murmurhash3_32(element, seed=2 ** 20 - 1, positive=True) % self.R)
            elif i == 3:
                templist.append(murmurhash3_32(element, seed=2 ** 25 - 1, positive=True) % self.R)
        return templist


def plotError_Freq(Freq, token_dict):
    for R in [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18]:
        x_list = []
        y_list_sketch = []
        y_list_minsketch = []
        sketch = CountSketch(4, R)
        minsketch = CountMinSketch(4, R)
        for key in token_dict.keys():
            sketch.add(key)
            minsketch.add(key)
        for key, actual_value in Freq.items():
            y_list_sketch.append(abs(sketch[key] - actual_value) / actual_value)
            y_list_minsketch.append(abs(minsketch[key] - actual_value) / actual_value)
            x_list.append(key)

        drawPlot(x_list, y_list_sketch, y_list_minsketch)


def findInFreq(token_dict):
    inFreq = {}
    minpq = PriorityQueue()
    for token, val in token_dict.items():
        minpq.put((val, token))
    counter = 100
    while not minpq.empty() and counter > 0:
        temp = minpq.get()
        inFreq[temp[1]] = temp[0]
        counter -= 1
    return inFreq


def findMostFreq(token_dict):
    Freq = {}
    minpq = PriorityQueue()
    for token, val in token_dict.items():
        minpq.put((-val, token))
    counter = 100
    while not minpq.empty() and counter > 0:
        temp = minpq.get()
        Freq[temp[1]] = -temp[0]
        counter -= 1
    return Freq


def findRandomFreq(token_dict):
    randFreq = {}
    for idx in range(100):
        key = choice(list(token_dict.keys()))
        randFreq[key] = token_dict[key]
    return randFreq


def drawPlot(x_list, y_list1, y_list2):
    plt.figure(figsize=(25, 20))
    plt.xlabel('Words', fontsize=24)
    plt.ylabel('Relative  Error', fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    plt.plot(x_list, y_list1, "x-", label="Count_sketch")
    plt.plot(x_list, y_list2, "+-", label="Count_min_sketch")
    plt.xticks(rotation=-90)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0., fontsize=20)
    plt.show()

def main():
    data = pd.read_csv('user-ct-test-collection-01.txt', encoding='utf-8', sep="\t")
    #data = pd.read_csv('test.txt', encoding='utf-8', sep="\t")
    querys = data["Query"].dropna()
    token_dict = {}
    for token in querys:
        arr = token.split(' ')
        for a in arr:
            if a not in token_dict:
                token_dict[a] = 1
            else:
                token_dict[a] += 1
    inFreq = findInFreq(token_dict)
    Freq = findMostFreq(token_dict)
    randFreq = findRandomFreq(token_dict)
    plotError_Freq(inFreq, token_dict)
    plotError_Freq(Freq, token_dict)
    plotError_Freq(randFreq, token_dict)

    """
    sketches with heaps
    Freq has contains actual top-100

    TODO: maintain a top-1000 coutns in a minheap
    """

    x_list = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18]
    y_list = []
    y_list1 = []
    dataset = []
    for query in querys:
        arr = query.split(' ')
        for a in arr:
            dataset.append(a)
    for R in x_list:
        # heapdict + dict来做. dict充当minheap， heapdict充当minheap里面的计数
        sketch = CountSketch(4, R)
        # min heap
        dict = {}

        hd = heapdict.heapdict()
        for token in dataset:
            sketch.add(token)

            """如果存在minheap中，那么就更新它的次数"""
            if token in dict.keys():
                dict[token] += 1
                hd[token] = dict[token]
            else:
                """如果元素不在minheap中，那么下面继续判断数量是否超过1000"""
                if len(dict) < 1000:
                    # 直接加入到minheap中
                    dict[token] = 1
                    hd[token] = dict[token]
                else:
                    """the number of elements in minheap is more than 1000"""
                    est_val = sketch[token]
                    minElem = hd.popitem()
                    if est_val < minElem[1]:
                        # keep the same
                        dict[minElem[0]] = minElem[1]
                        hd[minElem[0]] = minElem[1]
                    else:
                        """have to update the minheap"""
                        #fixed the bug, in the beginning, forget to delete element in minheap
                        dict.pop(minElem[0])

                        dict[token] = est_val
                        hd[token] = dict[token]
        count = 0
        for key, value in dict.items():
            if key in Freq.keys():
                count += 1

        y_list.append(count)

    for R in x_list:
        minsketch = CountMinSketch(4, R)
        # min heap
        dict = {}
        hd = heapdict.heapdict()
        for token in dataset:
            minsketch.add(token)
            """如果存在minheap中，那么就更新它的次数"""
            if token in dict.keys():
                dict[token] += 1
                hd[token] = dict[token]
            else:
                if len(dict) < 1000:
                    # 直接加入到minheap中
                    dict[token] = 1
                    hd[token] = dict[token]
                else:
                    """elements in minheap are more than 1000"""
                    est_val = minsketch[token]
                    minElem = hd.popitem()
                    if est_val < minElem[1]:
                        # keep the same
                        dict[minElem[0]] = minElem[1]
                        hd[minElem[0]] = minElem[1]
                    else:
                        """have to update the minheap"""
                        #fixed the bug, in the beginning, forget to delete element in minheap
                        dict.pop(minElem[0])
                        dict[token] = est_val
                        hd[token] = dict[token]
        count = 0
        for key, value in dict.items():
            if key in Freq.keys():
                count += 1
                # print(count)
        y_list1.append(count)
    x_list_str = []
    for x in x_list:
        x_list_str.append(str(x))
    plt.figure(figsize=(10,10))
    plt.xlabel('R-value', fontsize=24)
    plt.ylabel('Intersection Size', fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    plt.plot(x_list_str, y_list, "x-", label="Count_sketch")
    plt.plot(x_list_str, y_list1, "+-", label="Count_Min_sketch")
    plt.legend(loc = 'lower right')
    plt.show()

if __name__ == '__main__':
    main()
