import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.murmurhash import murmurhash3_32

P = 1048573
def hash_func(x, a, b, c, d, k):
    if k == 0:
        return (a * x + b) % P % 1024
    if k == 1:
        return (a * pow(x,2) + b * x + c) % P % 1024
    if k == 2:
        return (a * pow(x,3) + b * x * x + c * x + d) % P % 1024
    if k == 3:
        return murmurhash3_32(x) % 1024
    return -1
def flip(x, j):
    return x ^ (1 << j)

def main():
    # a, b, c, d uniformly between [1-1048573]
    a = random.randint(1, P + 1)
    b = random.randint(1, P + 1)
    c = random.randint(1, P + 1)
    d = random.randint(1, P + 1)
    x_list = []
    y_list = []
    for i in range(5000):
        x = random.randint(0, 1 << 31 - 1)
        x_list.append(x)
        ym = []
        for m in range(4):
            ym.append(hash_func(x, a, b, c, d, m))
        y_list.append(ym)

    data = []
    for m in range(4):
        # calculate p
        p = []
        for i in range(31):
            pi = []
            for j in range(10):
                pij = 0
                for k in range(5000):
                    x_flipped = flip(x_list[k], i)
                    y_flipped = hash_func(x_flipped, a, b, c, d, m)
                    diff = y_list[k][m] ^ y_flipped
                    pij += ((diff >> j) & 1)
                pij /= 5000
                pi.append(pij)
            p.append(pi)

        data.append(np.array(p).T)

    fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axn.flat):
        sns.heatmap(data[i], ax=ax, center=0.5)
    plt.savefig('test_hash_function.png', format='png')
    plt.show()

if __name__ == '__main__':
    main()