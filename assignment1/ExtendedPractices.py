import pandas as pd
import random
from assignment1 import Question3
import matplotlib.pyplot as plt
import sys
import math


data = pd.read_csv('user-ct-test-collection-01.txt', sep ="\t")
urllist = data.ClickURL.dropna().unique()

#sample 1000 urls from urllist
test_set = random.sample(urllist.tolist(), 2000)
#print(test_set)

membership_set = list(set(urllist) - set(test_set))
R_bits = []
res_fp = []
memory = []
for fp in [0.1,0.05, 0.03, 0.01, 0.007, 0.004, 0.001]:
    bloomfilter = Question3.Bloom_Filter(falsePosRate= fp, expectedKeys=377871, param=0.7)

    #adjust bloomfilter.R in range of power of 2
    temp = math.log(bloomfilter.R,2)
    bloomfilter.R = round(math.pow(2,temp))
    R_bits += [bloomfilter.R]

    for i in range(len(membership_set)):
        bloomfilter.insert(membership_set[i])

    #memory usage of a object can be check with sys.getsizeof()
    memory += [sys.getsizeof(bloomfilter.bit_array.tobytes())]
    ans = 0
    for item in test_set:
        if bloomfilter.test(item):
            ans += 1
    res_fp += [ans / 2000]

plt.plot(R_bits, res_fp, color="r", linestyle="--", marker="*", linewidth=1.0)
plt.xlabel('R_bits')
plt.ylabel('false positive rate')
plt.savefig('fig1', format='png')
plt.show()



plt.plot(R_bits, memory, color="r", linestyle="--", marker="*", linewidth=1.0)
plt.xlabel('R_bits')
plt.ylabel('memory usage')
plt.savefig('fig2', format='png')
plt.show()






