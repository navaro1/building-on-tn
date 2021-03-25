import uuid

import numpy as np
import tensornetwork as tn

from network_science.ns import nodes_equal


# TODO:
# - add parsing from adjacency matrix
# - add returning an adjacency matrix
# - remember about undirected network

class DirectedNetwork:
    """Class representing Networks. Network is a collection of nodes joined by edges"""

    def __init__(self, adjacency_matrix: np.ndarray = None) -> None:
        self._nodes = {}
        self._edges = {}
        if adjacency_matrix is None or adjacency_matrix.shape[-1] == 0:
            pass
        else:
            if not all(np.array(adjacency_matrix.shape) == adjacency_matrix.shape[0]):
                raise ValueError(f"adjacency matrix needs to be square, but was {adjacency_matrix.shape}")
            it = np.nditer(adjacency_matrix, flags=['multi_index'])
            for value in it:
                if value == 0:
                    for idx in it.multi_index:
                        self.add_node_if_does_not_exist(0, idx + 1)  # shift is required to enumerate nodes from 1
                elif all(np.array(it.multi_index) == it.multi_index[0]):  # diagonal
                    if value % 2 != 0:
                        raise ValueError("Non even number on an diagonal")
                    node_idx = it.multi_index[0] + 1
                    self.add_node_if_does_not_exist(0, node_idx)
                    for times in range(value / 2):
                        self.add_edge(node_idx, node_idx)
                else:  # Not handling hypergraphs as of now
                    idx_from = it.multi_index[0]
                    idx_to = it.multi_index[1]
                    self.add_node_if_does_not_exist(0, idx_from)
                    self.add_node_if_does_not_exist(0, idx_to)
                    self.add_edge(idx_from, idx_to)

    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def edges(self):
        return self._edges.copy()

    def create_nodes_and_edge(self, from_value, to_value, from_name: str = None, to_name: str = None) -> None:
        self.add_node_if_does_not_exist(from_value)
        self.add_node_if_does_not_exist(to_value)
        str_from = from_name if from_name is not None else str(from_value)
        str_to = to_name if to_name is not None else str(to_value)
        if str_from not in self._edges or str_to not in self._edges[str_from]:
            self.add_edge(str_from, str_to)

    def add_node(self, node_value, node_name: str = None) -> None:
        actual_node_name = node_name if node_name is not None else str(node_value)
        if actual_node_name in self._nodes:
            raise ValueError('Node: "%s" already exists' % actual_node_name)

        new_node = tn.Node(node_value, name=actual_node_name)
        self._nodes[actual_node_name] = new_node

    def add_node_if_does_not_exist(self, node_value, node_name: str = None) -> None:
        actual_node_name = node_name if node_name is not None else str(node_value)
        if actual_node_name not in self._nodes:
            new_node = tn.Node(node_value, name=actual_node_name)
            self._nodes[actual_node_name] = new_node

    def add_node_unique(self, node_value) -> str:
        node_name = str(uuid.uuid4())
        self.add_node(node_value, node_name)
        return node_name

    def remove_node(self, node_name) -> bool:
        result = False
        str_node_name = str(node_name)
        if str_node_name in self._nodes:
            self._nodes.pop(str_node_name)
            result = True
            for key in self._edges.keys():
                if key == str_node_name:
                    self._edges.pop(str_node_name)
                else:
                    nodes = self._edges[key]
                    self._edges[key] = list(filter(lambda a: a != str_node_name, nodes))
        return result

    def add_edge(self, from_node, to_node) -> None:
        str_from = str(from_node)
        str_to = str(to_node)
        if str_from not in self._nodes:
            raise ValueError('Node: "%s" does not exist' % str_from)
        if str_to not in self._nodes:
            raise ValueError('Node: "%s" does not exist' % to_node)
        if str_from not in self._edges:
            self._edges[str_from] = list(str_to)
        else:
            self._edges[str_from].append(str_to)

    def remove_edge(self, from_node, to_node) -> bool:
        str_from = str(from_node)
        str_to = str(to_node)
        result = False
        if str_from in self._edges:
            try:
                current_edges = self._edges[str_from]
                current_edges.remove(str_to)
                if len(current_edges) == 0:
                    self._edges.pop(str_from)
                result = True
            except ValueError:
                result = False
        return result

    def __str__(self) -> str:
        node_to_children = [key + "->" + value
                            for (key, values) in self._edges.items()
                            for value in values]
        return "\n".join(node_to_children)

    def __eq__(self, other):
        return self._edges == other.edges and self._compare_all_nodes(other)

    def _compare_all_nodes(self, other):
        # Very inefficient
        for key in self._nodes.keys():
            if key not in other.nodes:
                return False
            self_node = self._nodes[key]
            other_node = other.nodes[key]
            if not nodes_equal(self_node, other_node):
                return False
        return True


class UndirectedNetwork(DirectedNetwork):

    def create_nodes_and_edge(self, from_value, to_value, from_name: str = None, to_name: str = None) -> None:
        super().create_nodes_and_edge(from_value, to_value, from_name, to_name)
        super().add_edge(to_value, from_value)

    def add_node(self, node_value, node_name: str = None) -> None:
        super().add_node(node_value, node_name)

    def add_node_unique(self, node_value) -> str:
        return super().add_node_unique(node_value)

    def remove_node(self, node_name) -> bool:
        return super().remove_node(node_name)

    def add_edge(self, from_node, to_node) -> None:
        super().add_edge(from_node, to_node)
        super().add_edge(to_node, from_node)

    def remove_edge(self, from_node, to_node) -> bool:
        return super().remove_edge(from_node, to_node) and super().remove_edge(to_node, from_node)

    def __str__(self) -> str:
        return super().__str__()
