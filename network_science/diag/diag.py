import numpy as np
from numpy.linalg import eig
from mod import Mod
from scipy.sparse import csgraph

dim = 5
z = np.zeros([dim, dim])
upto = (dim - 1) // 2
multiplier = 3
dm = {(multiplier * x) % dim for x in set(range(1, 1 + upto, 1))}
print(dm)
it = np.nditer(z, flags=['multi_index'])
for i in range(dim):
    for j in range(dim):
        m = Mod(j - i, dim)
        if m in dm:
            # print('%s testuje %s' % (i, j))
            z[i][j] = 1
np.random.seed(24)
random_mtrx = np.random.choice([0, 1], size=(dim, dim), p=[1. / 2, 1. / 2])
# test_result = z * random_mtrx
# test_result = np.array([
#     [0, 1, 1, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1],
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
# ])
unos = np.ones(shape=(dim, dim))
print(z)
# print(test_result)
eigval_u, eigvec_u = eig(unos)
eigval_z, eigvec_z = eig(z)
print("=============")
print("=============")
print(eigval_u)
print(np.poly1d(np.poly(unos)))
print("=============")
print(eigval_z)
print(np.poly1d(np.poly(z)))
# print(eigvec)
# # print(eigvec[4])
# # print(np.linalg.norm(eigvec[4]))
print("=============")
l_u = csgraph.laplacian(unos, normed=False)
l_z = csgraph.laplacian(z, normed=False)
# print(l)
print("====== BIGGU =======")
eigval_l, eigvec_l = eig(l_u)
print(eigval_l)
print(np.poly1d(np.poly(l_u)))
print("===== SMOL =======")
eigval_lz, eigvec_lz = eig(l_z)
print(eigval_lz)
print(np.poly1d(np.poly(l_z)))
print("=============")
