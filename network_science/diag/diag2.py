import numpy as np
from numpy.linalg import eig
from mod import Mod
from scipy.sparse import csgraph


dim = 3
z = np.zeros([dim, dim])
upto = (dim - 1) // 2
dm = set(range(1, 1 + upto, 1))

it = np.nditer(z, flags=['multi_index'])
for i in range(dim):
    for j in range(dim):
        m = Mod(j - i, dim)
        if m in dm:
            # print('%s testuje %s' % (i, j))
            z[i][j] = 1
np.random.seed(24)
random_mtrx = np.random.choice([0, 1], size=(dim, dim), p=[1. / 2, 1. / 2])
print(z)
print(random_mtrx)

