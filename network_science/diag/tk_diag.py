from itertools import chain, combinations, permutations, islice, combinations_with_replacement
from multiprocessing import Pool
import more_itertools

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from mod import Mod
from scipy.sparse import csgraph
from numpy.linalg import eig

np.random.seed(1302)
dim = 20
rndm_mtrx = np.random.choice([0, 1], size=(dim, dim), p=[1. / 2, 1. / 2])
z = np.zeros([dim, dim])
upto = (dim - 1) // 2
multiplier = 1
dm = {(multiplier * x) % dim for x in set(range(1, 1 + upto, 1))}

it = np.nditer(z, flags=['multi_index'])
for i in range(dim):
    for j in range(dim):
        m = Mod(j - i, dim)
        if m in dm:
            # print('%s testuje %s' % (i, j))
            z[i][j] = 1
print(rndm_mtrx)
print("=============")
print(z)
print("=============")


def vertices(square_matrix: np.array):
    return list(range(len(square_matrix)))


verts = vertices(rndm_mtrx)

covers_map = {}


def covers(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in covers_wrapper(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def covers_wrapper(collection):
    s = str(collection)
    if s not in covers_map:
        covers_map[s] = list(covers(collection))
    return covers_map[s]


def covers_wth_le_than_t_elements(covers, t=1):
    result = []
    for cover in covers:
        if all(len(c) <= t for c in cover):
            result.append(cover)
    return result


def covers_with_more_than_k_shared_elements(cover, k=1):
    result = set(cover[0])
    for element in cover[1:]:
        result = result.intersection(set(element))
    return len(result) >= k


def f_dash_greater_than_k(verts, vert_subset, k=1):
    return len(set(verts).difference(set(vert_subset))) >= k


def out_nodes(vertex, matrix):
    it = np.nditer(matrix[vertex], flags=['multi_index'])
    return [it.multi_index[0] for value in it if value == 1]


def in_nodes(vertex, matrix):
    it = np.nditer(matrix[:, vertex], flags=['multi_index'])
    return [it.multi_index[0] for value in it if value == 1]


def edges(matrix):
    it = np.nditer(matrix, flags=['multi_index'])
    return [it.multi_index for value in it if value == 1]


def elements_of_cover_which_belongs(vertex, cover):
    return {tuple(c) for c in cover if vertex in c}


def edge_condition(edges, cover):
    cover_set = {tuple(i) for i in cover}
    for edge in edges:
        u = edge[0]
        v = edge[1]

        fu = elements_of_cover_which_belongs(u, cover)
        fv = elements_of_cover_which_belongs(v, cover)
        if not fu.issubset(fv) and fu.union(fv) != cover_set:
            return True
    return False


def non_empty_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def subsets_of_size(iterable, size):
    s = list(iterable)
    return combinations(s, size)


# def is_tk_diagnosable(matrix, t, k):
#     print(matrix)
#     vs = vertices(matrix)
#     edgs = edges(matrix)
#     for subset_f in non_empty_powerset(vs):
#         cvs = covers_wrapper(list(subset_f))
#         good_cvs = covers_wth_le_than_t_elements(cvs, t)
#         for good_cv in good_cvs:
#             # cond 4
#             if len(good_cv) == 1:
#                 continue
#             # cond 3
#             if edge_condition(edgs, good_cv):
#                 continue
#             # cond 1
#             if covers_with_more_than_k_shared_elements(good_cv, k):
#                 continue
#             # cond 2
#             if out_degree_greater_than_k(matrix, subset_f, vs, k):
#                 continue
#             return False
#     return True


def out_degree_greater_than_k(matrix, subset_f, vs, k):
    f_dash = set(vs).difference(set(subset_f))
    out_res = set()
    for v in f_dash:
        out_res = out_res.union(set(out_nodes(v, matrix)).difference(f_dash))
    return len(out_res) >= k


vert_to_in_nodes = {}


def in_degree(matrix, subset_u):
    in_res = set()
    for v in subset_u:
        if v not in vert_to_in_nodes:
            vert_to_in_nodes[v] = set(in_nodes(v, matrix))
        in_res = in_res.union(vert_to_in_nodes[v].difference(subset_u))
    return in_res


bad_example = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0]]
)


def other_is_tk_diagnosable(matrix, t, k):
    if len(matrix) - 1 < 2 * t:
        return False
    vs = vertices(matrix)
    if not all([len(in_nodes(v, matrix)) >= k for v in vs]):
        return False
    for p in reversed(range(1, k)):
        sub_us = subsets_of_size(vs, 2 * (t - p))
        for sub_u in sub_us:
            if not len(in_degree(matrix, sub_u)) > p:
                return False
    return True


print("=!=!=!=!=")


# print(other_is_tk_diagnosable(z, t=9, k=9))

def check_if_t_k_diagnosable(c, t, k):
    mtrx = np.array(c)
    if np.trace(mtrx) > 0:
        return None
    if other_is_tk_diagnosable(mtrx, t, k):
        return mtrx


def generate_all_minimal_t_k_matrices_of_dim(dim, t, k):
    if dim < 2 * t + 1:
        yield
        return
    if k > t:
        yield
        return
    base = [0] * (dim - k) + [1] * k
    perms = set(permutations(base, dim))
    inputs = permutations(perms, dim)
    for i in inputs:
        mtrx = np.array(i)
        if np.trace(mtrx) == 0 and other_is_tk_diagnosable(mtrx, t, k):
            yield mtrx


def map_to_eigens(m):
    l_z = csgraph.laplacian(m, normed=True)
    return np.poly(l_z).round(13)


num_cores = multiprocessing.cpu_count()
for n in range(3, 20):
    s = []
    print("Working on " + str(n))
    tk = (n - 1) // 2
    res = tqdm(generate_all_minimal_t_k_matrices_of_dim(n, 1, 1))
    for chunk in more_itertools.chunked(res, 100_000):
        temp = Parallel(
            n_jobs=num_cores - 1)(delayed(map_to_eigens)(i) for i in tqdm(chunk))
        uniqs = np.vstack(list({tuple(row) for row in temp}))
        s.append(uniqs)
    print("DONE PART UNO!")
    if len(s) > 0:
        s = np.vstack(s)

    set_of_polys = set({str(e) for e in s})
    with open("tk_polys_3.txt", "a") as f:
        f.write("========= " + str(n) + " =========\n")
        f.write(str(len(set_of_polys)))
        f.write("\n")
        for a in set_of_polys:
            f.write(a)
            f.write("\n")

# print(d)
