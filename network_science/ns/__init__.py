import tensornetwork as tn


def nodes_equal(node_a: tn.Node, node_b: tn.Node) -> bool:
    return node_a.tensor == node_b.tensor \
           and node_a.name == node_b.name \
           and node_a.edges == node_b.edges \
           and node_a.shape == node_b.shape \
           and node_a.axis_names == node_b.axis_names \
           and node_a.dtype == node_b.dtype \
           and node_a.sparse_shape == node_b.sparse_shape \
           and node_a.__class__ == node_b.__class__
