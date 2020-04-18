from nltk.util import ngrams
from sklearn.utils import murmurhash3_32
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv
def taskZero():
    s1 = "The mission statement of the WCSCC and area employers recognize the importance of good attendance on the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for advanced placement as well as hindering his/her likelihood for successfully completing their program."
    s2 = "The WCSCC’s mission statement and surrounding employers recognize the importance of great attendance. Any student who is absent more than 18 days will loose the opportunity for successfully completing their trade program."
    ngrams1 = trans_ngrams(s1,3)
    ngrams2 = trans_ngrams(s2,3)
    real_jaccard = cal_jaccard(set(ngrams1),set(ngrams2))
    print("Real jaccard similarity for two strings are", real_jaccard)

    min_hash1 = MinHash(ngrams1,100)
    min_hash2 = MinHash(ngrams2,100)
    estimated_jaccard = cal_jaccard(set(min_hash1), set(min_hash2))
    print("Estimated jaccard similarity for two strings are", estimated_jaccard)

def taskOne(urllist):
    K = 2
    L = 50
    R = 2**20
    ht = HashTable(K, L, R)
    jaccard_avg = []
    jaccard_avg_top10 = []

    for url in urllist:
        #insert all urls into minhash hashtables
        ht.insert(MinHash(url,K*L),url)
    query_list = np.random.choice(urllist,200)
    #save query_list into file
    filename = 'Sample_url.txt'
    with open(filename, 'w') as f:
        for url in query_list:
            f.write(url)
            f.write('\n')
    totaltime = 0
    for i in range(len(query_list)):
        start_time = time.time()
        simItems = ht.lookup(MinHash(query_list[i], K*L))
        len_items = len(simItems)
        temp = np.zeros(len_items)
        #we only care about the similar item with query_set
        for j in range(len_items):
            temp[j] = cal_jaccard(set(simItems[j]), set(query_list[i]))

        jaccard_avg.append(np.average(temp))
        end_time = time.time()
        totaltime = totaltime + (end_time - start_time)
        jaccard_avg_top10.append(-np.average(np.partition(-temp,10)[:10]))
    print("The minhash method for total query time is : ", totaltime, 'seconds')
    sava_data(jaccard_avg,jaccard_avg_top10)

    """TASK 2, compute the pairwise Jaccard similarity of those 200 URLs in the query set"""
    start_time2 = time.time()
    pairwise_jaccard = np.zeros([len(query_list), len(urllist)])
    for i in range(len(query_list)):
        for j in range(len(urllist)):
            #忘记转化为3-gram
            pairwise_jaccard[i][j] = cal_jaccard(set(trans_ngrams(query_list[i],3)),set(trans_ngrams(urllist[j],3)) )
    end_time2 = time.time()
    run_time = end_time2 - start_time2
    print("The brute force for per query  time is ", run_time / len(query_list), 'seconds')
    print("The brute force for total query time is ", (run_time / len(query_list)) * len(urllist), 'seconds', ' that is ',((run_time / len(query_list)) * len(urllist))/3600, 'days' )

def taskThree(urllist):
    for K in [2,3,4,5,6]:
        for L in [20,50,100]:
            ht = HashTable(K, L, 2**10, 2**10)
            counter = 0
            for url in urllist:
                counter += 1
                if(counter > 50000):
                    break
                ht.insert(MinHash(url, K * L), url)
            jaccard_avg = []
            query_list = np.random.choice(urllist, 200)
            start_time = time.time()
            for i in range(len(query_list)):
                items = ht.lookup(MinHash(query_list[i], K * L))
                len_items = len(items)
                temp = np.zeros(len_items)
                for j in range(len_items):
                    temp[j] = cal_jaccard(set(items[j]), set(query_list[i]))
                jaccard_avg.append(np.average(temp))
            end_time = time.time()
            print('When K = {}, L  = {}'.format(K, L))
            print("The time per query is : ", (end_time - start_time) / 200)

def taskFour():
    f2 = np.vectorize(f)
    x = np.arange(0, 1, 0.01)
    plt.gca().set_prop_cycle(color = ['red','orange','gold','yellow','green','blue','purple'])
    for K in [1,2,3,4,5,6,7]:
        plt.plot(x,f2(x,K))
    plt.xlabel('Jaccard similarity with query', fontsize=14)
    plt.ylabel('Probability of Retrieving', fontsize=14)
    plt.legend(['K=1', 'K=2', 'K=3', 'K=4','K=5','K=6','K=7'], loc='lower right')
    plt.show()

    g2 = np.vectorize(g)
    plt.gca().set_prop_cycle(color = ['red','orange','gold','yellow','green','blue','purple'])
    for L in [5,10,20,50,100,150,200]:
        plt.plot(x,g2(x,L))
    plt.xlabel('Jaccard similarity with query', fontsize=14)
    plt.ylabel('Probability of Retrieving', fontsize=14)
    plt.legend(['L=5', 'L=10', 'L=20', 'L=50','L=100','L=150','L=200'], loc='lower right')
    plt.show()

def f(x,K):
    return 1 - math.pow(1 - math.pow(x, K), 50)
def g(x,L):
    return 1 - math.pow(1 - math.pow(x, 4), L)

class HashTable():
    def __init__(self, K, L, R):
        """
        :param K:  number of hash functions
        :param L:  number of hash tables
        :param R:  the number of buckets 2**20
        """
        self.K = K
        self.L = L
        self.R = R
        self.tables = [[[] for _ in range(self.R)] for _ in range(self.L)]

    def insert(self, hashcodes, id):
        """
        The function is to insert this id into a different buckets in each hash table according to its hashcodes"
        tables[i][j]
        """
        for i in range(0,self.K * self.L, self.K):
            tablecIdx = int(i / self.K)
            depthIdx = hash(str(hashcodes[i:i+self.K])) % self.R
            self.tables[tablecIdx][depthIdx].append(id)

        # for i in range(0,self.K * self.L, self.K):
        #     b = math.pow(self.B, 1/self.K)
        #     idx = 0
        #     for j in range(self.K):
        #         idx = int(idx * b + hashcodes[i+j] % b)
        #     self.tables[int(i / self.K)][idx].append(id)



    def lookup(self, hashcodes):
        """
        The function is to retrieve all the items in the hash tables according to the supplied hashcodes
        """
        # res_list = []
        # for i in range(0,self.K * self.L, self.K):
        #     b = math.pow(self.B, 1/self.K)
        #     idx = 0
        #     for j in range(self.K):
        #         idx = int(idx * b + hashcodes[i+j] % b)
        #         if len(self.tables[int(i/self.K)][idx]) > 0:
        #             res_list += self.tables[int(i/self.K)][idx]
        # return list(set(res_list))

        res_list = []
        for i in range(0, self.K * self.L, self.K):
            tablecIdx = int(i / self.K)
            depthIdx = hash(str(hashcodes[i:i + self.K])) % self.R
            bucket_list = self.tables[tablecIdx][depthIdx]
            if(len(bucket_list) > 0):
                res_list.extend(bucket_list)
        return list(set(res_list))


def trans_ngrams(str,n):
    n_grams = ngrams(str, n)
    return [''.join(grams) for grams in n_grams]
"""
Learned from lecture
Document : S =  {ama, maz, azo, zon, on.}.
Generate Random 𝑈_𝑖:𝑆𝑡𝑟𝑖𝑛𝑔𝑠→𝑁. Example: Murmurhash3 with new random seed i. 
 𝑈_𝑖 (𝑆)"={" 𝑈_𝑖 "(ama)," 〖 𝑈〗_𝑖 "(maz)," 〖 𝑈〗_𝑖 "(azo)," 〖 𝑈〗_𝑖 "(zon)," 〖 𝑈〗_𝑖 "(on.)}"
 Lets say 𝑈_𝑖 (𝑆)"=" {153, 283, 505, 128, 292}
 Then Minhash:  ℎ_𝑖 = min ℎ_𝑖 (𝑆) = 128. 
"""
def MinHash(A, m):
    """
    :param A:  the input string
    :param m:  the number of hash functions
    :return:
    """
    min_hash = []
    for i in range(m):
        temp = []
        for j in range(len(A)):
            temp.append(murmurhash3_32(A[j],seed = i))
        min_hash.append(min(temp))
    #return m-length hashcodes
    return min_hash

def cal_jaccard(set1, set2):
    """compute the jaccard similarity between two sets"""
    return float(len(set1 & set2)) / float(len(set1 | set2))

def sava_data(jaccard_avg,jaccard_av1_top10):
    filename1 = 'Task1:jaccard similarity.txt'
    with open(filename1, 'w') as f:
        f.write("The mean jaccard similarity of URL retrieved:\n")
        for j in jaccard_avg:
            f.write(str(j))
            f.write('\n')
    filename2 = 'Task1:top-10 jaccard similarity.txt'
    with open(filename2, 'w') as f:
        f.write("The top 10 urls for their mean jaccard similarity of URL retrieved:\n")
        for j in jaccard_av1_top10:
            f.write(str(j))
            f.write('\n')

def main():
    """TASK 0"""
    #taskZero()
    #data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
    #data = pd.read_csv('test.txt', sep="\t")
    #urllist = data.ClickURL.dropna().unique()
    """TASK 1,2"""
    #taskOne(urllist)

    """TASK 3"""
    #taskThree(urllist)

    """TASK 4"""
    taskFour()



if __name__ == '__main__':
    main()