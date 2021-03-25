import uuid

import numpy as np
import tensornetwork as tn


# - improve string representation

class DirectedNetwork:
    """Class representing Networks. Network is a collection of nodes joined by edges"""

    def __init__(self, adjacency_matrix: np.ndarray = None) -> None:
        self._nodes = {}
        self._edges = {}
        self._adjacency_matrix = adjacency_matrix
        if adjacency_matrix is None or adjacency_matrix.shape[-1] == 0:
            pass
        else:
            self._build_from_adj_matrix(adjacency_matrix)

    def _build_from_adj_matrix(self, adjacency_matrix):
        if not all(np.array(adjacency_matrix.shape) == adjacency_matrix.shape[0]):
            raise ValueError(f"adjacency matrix needs to be square, but was {adjacency_matrix.shape}")
        it = np.nditer(adjacency_matrix, flags=['multi_index'])
        for value in it:
            if value == 0:
                for idx in it.multi_index:
                    self.add_node_if_does_not_exist(0, str(idx + 1))  # shift is required to enumerate nodes from 1
            elif all(np.array(it.multi_index) == it.multi_index[0]):  # diagonal
                if value % 2 != 0:
                    raise ValueError("Non even number on an diagonal")
                node_idx = it.multi_index[0] + 1
                self.add_node_if_does_not_exist(0, str(node_idx))
                for times in range(value // 2):
                    self.add_edge(node_idx, node_idx)
            else:  # Not handling hypergraphs as of now
                for idx in range(value):
                    idx_from = str(it.multi_index[0] + 1)
                    idx_to = str(it.multi_index[1] + 1)
                    self.add_node_if_does_not_exist(0, idx_from)
                    self.add_node_if_does_not_exist(0, idx_to)
                    self.add_edge(idx_from, idx_to)

    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def adjacency_matrix(self):
        # For now discard values
        if self._adjacency_matrix is None:
            size = len(self._nodes)
            node_name_to_idx = {}
            idx = 0
            for node in self._nodes.keys():
                node_name_to_idx[node] = idx
                idx += 1
            adj_matrix = np.zeros(shape=(size, size))  # only 2d
            for node in self._nodes.keys():
                if node in self._edges:
                    for target_node in self._edges[node]:
                        from_idx = node_name_to_idx[node]
                        target_idx = node_name_to_idx[target_node]
                        adj_matrix[from_idx][target_idx] = adj_matrix[from_idx][target_idx] + (
                            2 if from_idx == target_idx else 1)
            self._adjacency_matrix = adj_matrix
        return self._adjacency_matrix

    def create_nodes_and_edge(self, from_value, to_value, from_name: str = None, to_name: str = None) -> None:
        self.add_node_if_does_not_exist(from_value)
        self.add_node_if_does_not_exist(to_value)
        str_from = from_name if from_name is not None else str(from_value)
        str_to = to_name if to_name is not None else str(to_value)
        if str_from not in self._edges or str_to not in self._edges[str_from]:
            self._adjacency_matrix = None  # Refresh adj matrix on next retrieval
            self.add_edge(str_from, str_to)

    def add_node(self, node_value, node_name: str = None) -> None:
        actual_node_name = node_name if node_name is not None else str(node_value)
        if actual_node_name in self._nodes:
            raise ValueError('Node: "%s" already exists' % actual_node_name)

        new_node = tn.Node(node_value, name=actual_node_name)
        self._adjacency_matrix = None  # Refresh adj matrix on next retrieval
        self._nodes[actual_node_name] = new_node

    def add_node_if_does_not_exist(self, node_value, node_name: str = None) -> None:
        actual_node_name = node_name if node_name is not None else str(node_value)
        if actual_node_name not in self._nodes:
            new_node = tn.Node(node_value, name=actual_node_name)
            self._nodes[actual_node_name] = new_node

    def add_node_unique(self, node_value) -> str:
        node_name = str(uuid.uuid4())
        self.add_node(node_value, node_name)
        self._adjacency_matrix = None  # Refresh adj matrix on next retrieval
        return node_name

    def remove_node(self, node_name) -> bool:
        result = False
        str_node_name = str(node_name)
        if str_node_name in self._nodes:
            self._nodes.pop(str_node_name)
            result = True
            self._adjacency_matrix = None  # Refresh adj matrix on next retrieval
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
        self._adjacency_matrix = None  # Refresh adj matrix on next retrieval
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
                self._adjacency_matrix = None  # Refresh adj matrix on next retrieval
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
        return (self.adjacency_matrix == other.adjacency_matrix).all()


class UndirectedNetwork(DirectedNetwork):

    def __init__(self, adjacency_matrix: np.ndarray = None) -> None:
        if adjacency_matrix is not None and \
                (not all(np.array(adjacency_matrix.shape) == adjacency_matrix.shape[0])
                 or not (adjacency_matrix.transpose() == adjacency_matrix).all()):
            raise ValueError("Undirected Network can have only symmetric adjacency matrix")
        super().__init__(adjacency_matrix)

    def _build_from_adj_matrix(self, adjacency_matrix):
        it = np.nditer(adjacency_matrix, flags=['multi_index'])
        for value in it:
            if value == 0:
                for idx in it.multi_index:
                    self.add_node_if_does_not_exist(0, str(idx + 1))  # shift is required to enumerate nodes from 1
            elif all(np.array(it.multi_index) == it.multi_index[0]):  # diagonal
                if value % 2 != 0:
                    raise ValueError("Non even number on an diagonal")
                node_idx = it.multi_index[0] + 1
                self.add_node_if_does_not_exist(0, str(node_idx))
                for times in range(value // 2):
                    self.add_edge(node_idx, node_idx)
            else:
                # Not handling hypergraphs as of now
                for times in range(value):
                    idx_from = str(it.multi_index[0] + 1)
                    idx_to = str(it.multi_index[1] + 1)
                    self.add_node_if_does_not_exist(0, idx_from)
                    self.add_node_if_does_not_exist(0, idx_to)
                    super().add_edge(idx_from, idx_to)

    @property
    def adjacency_matrix(self):
        # For now discard values
        if self._adjacency_matrix is None:
            size = len(self._nodes)
            node_name_to_idx = {}
            idx = 0
            for node in self._nodes.keys():
                node_name_to_idx[node] = idx
                idx += 1
            adj_matrix = np.zeros(shape=(size, size))  # only 2d
            for node in self._nodes.keys():
                if node in self._edges:
                    for target_node in self._edges[node]:
                        from_idx = node_name_to_idx[node]
                        target_idx = node_name_to_idx[target_node]
                        adj_matrix[from_idx][target_idx] = adj_matrix[from_idx][target_idx] + 1
            self._adjacency_matrix = adj_matrix
        return self._adjacency_matrix

    def add_edge(self, from_node, to_node) -> None:
        super().add_edge(from_node, to_node)
        super().add_edge(to_node, from_node)

    def remove_edge(self, from_node, to_node) -> bool:
        return super().remove_edge(from_node, to_node) and super().remove_edge(to_node, from_node)

    def __str__(self) -> str:
        return super().__str__()

    def __eq__(self, other):
        return super().__eq__(other)
