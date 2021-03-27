import network_science.ns.network as n
import numpy as np
import copy

adj_matrix = np.array(
    [[0, 2, 0, 0, 1, 0],
     [1, 0, 1, 1, 0, 0],
     [0, 1, 4, 1, 1, 1],
     [0, 1, 1, 0, 0, 0],
     [1, 0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 2]])
dn_from_adj = n.DirectedNetwork(adj_matrix)

dn = n.DirectedNetwork()
for node in range(1, 7):
    dn.add_node(0, str(node))
dn.add_edge("1", "2")
dn.add_edge("1", "2")
dn.add_edge("1", "5")

dn.add_edge("2", "1")
dn.add_edge("2", "3")
dn.add_edge("2", "4")

dn.add_edge("3", "2")
dn.add_edge("3", "3")
dn.add_edge("3", "3")
dn.add_edge("3", "4")
dn.add_edge("3", "5")
dn.add_edge("3", "6")

dn.add_edge("4", "2")
dn.add_edge("4", "3")

dn.add_edge("5", "1")
dn.add_edge("5", "3")

dn.add_edge("6", "3")
dn.add_edge("6", "6")
assert dn_from_adj == dn
assert (adj_matrix == dn.adjacency_matrix).all()