from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import math
import random
import time
# implement hash function factory using murmurhash
# 3.1.1
def hashFunctionFactory(key,len):
    #use current time as seed
    t = time.time()
    seed = t * 1000000 % (math.floor(t) * 1000000)
    return murmurhash3_32(key = key, seed = seed) % len

#3.1.2
class Bloom_Filter:
    #constructor method.   Parameters: false positive rate, number of keys
    def __init__(self,falsePosRate, expectedKeys, param):
        self.falsePosRate = falsePosRate
        self.expectedKeys = expectedKeys
        #according to the formula in formulation, k = R/Nln2, and we know that R/N = math.log(f_p,0.618)
        self.k = math.floor(math.log(2) * math.log(self.falsePosRate, param))
        # R bits
        self.R = math.floor(math.log(self.falsePosRate,param) * self.expectedKeys)
        # adjust bloomfilter.R in range of power of 2
        temp = math.log(self.R, 2)
        self.R = round(math.pow(2, temp))

        # create a bitarray, 0 refers to false, 1 refers to true
        self.bit_array = bitarray(self.R)
        self.bit_array.setall(False)
    def __len__(self):
            return self.expectedKeys
    def __iter__(self):
        return iter(self.bit_array)

    def insert(self,key):
        for i in range(self.k):
            index = hashFunctionFactory(key,self.R)
            self.bit_array[index] = True
    def test(self, key):
        for i in range(self.k):
            index = hashFunctionFactory(key,self.R)
            if not self.bit_array[index]:
                return False
        return True

#3.1.3 test bloom filter with two datasets
def main():
    set1 = list(range(10000,100000))
    membership_set = random.sample(set1,10000)
    set2 = list(set(set1) - set(membership_set))
    #only pick 1000 from membership test but not int the membership set
    test_set = random.sample(set2,1000) + random.sample(set1,1000)
    for falsePositiveRate in [0.01, 0.001, 0.0001]:
        bf = Bloom_Filter(falsePositiveRate, 10000, 0.618)
        #insert
        for item in membership_set:
            bf.insert(item)
        res = 0
        for item1 in test_set:
            if bf.test(item1):
                res += 1
        ans = res / 2000;
        print('Theoretical FP = {}, Real FP = {:.8f}'.format(falsePositiveRate, ans))

if __name__ == '__main__':
        main()