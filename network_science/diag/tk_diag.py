from itertools import chain, combinations

import numpy as np
from mod import Mod

np.random.seed(1302)
dim = 3
rndm_mtrx = np.random.choice([0, 1], size=(dim, dim), p=[1. / 2, 1. / 2])
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
print(rndm_mtrx)
print("=============")
print(z)
print("=============")


def vertices(square_matrix: np.array):
    return list(range(len(square_matrix)))


verts = vertices(rndm_mtrx)


def covers(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in covers(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


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
    return [tuple(element) for element in cover if vertex in element]


def edge_condition(edges, cover):
    cover_set = {tuple(i) for i in cover}
    for edge in edges:
        u = edge[0]
        v = edge[1]
        fu = set(elements_of_cover_which_belongs(u, cover))
        fv = set(elements_of_cover_which_belongs(v, cover))
        if not fu.issubset(fv) and fu.union(fv) != cover_set:
            return True
    return False


def non_empty_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def is_tk_diagnosable(matrix, t, k):
    print(matrix)
    vs = vertices(matrix)
    edgs = edges(matrix)
    for subset_f in non_empty_powerset(vs):
        cvs = list(covers(subset_f))
        good_cvs = covers_wth_le_than_t_elements(cvs, t)
        for good_cv in good_cvs:
            # cond 2
            if out_degree_greater_than_k(matrix, subset_f, vs, k):
                continue
            # cond 3
            if edge_condition(edgs, good_cv):
                continue
            # cond 4
            if len(good_cv) == 1:
                continue
            # cond 1
            if covers_with_more_than_k_shared_elements(good_cv, k):
                continue
            return False
    return True


def out_degree_greater_than_k(matrix, subset_f, vs, k):
    f_dash = set(vs).difference(set(subset_f))
    out_res = set()
    for v in f_dash:
        out_res = out_res.union(set(out_nodes(v, matrix)).difference(f_dash))
    return len(out_res) >= k


bad_example = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0]]
)
print("=!=!=!=!=")
print(is_tk_diagnosable(z, t=5, k=5))
