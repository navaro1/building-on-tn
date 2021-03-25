import copy

import numpy as np
import pytest
import tensornetwork as tn

import network_science.ns.network as n
from network_science.ns import nodes_equal


def test_undirected_add_node_should_add_node():
    un = n.UndirectedNetwork()
    un.add_node(1)
    assert len(un.nodes) == 1
    assert nodes_equal(un.nodes['1'], tn.Node(1, name="1"))


def test_undirected_add_node_should_throw_if_node_already_exist():
    un = n.UndirectedNetwork()
    un.add_node(1)
    with pytest.raises(ValueError):
        un.add_node(1)


def test_adding_and_removing_node_should_leave_network_in_unchanged_state():
    un = n.UndirectedNetwork()
    un.add_node(1)
    copied_un = copy.deepcopy(un)
    un.add_node(2)
    removed = un.remove_node(2)
    assert removed is True
    assert un == copied_un


def test_removing_non_existing_node_should_throw_an_error():
    un = n.UndirectedNetwork()
    assert un.remove_node(2) is False


def test_add_edge_should_connect_two_nodes():
    un = n.UndirectedNetwork()
    un.add_node(1)
    un.add_node(2)
    un.add_edge(1, 2)
    assert un.edges['1'] == list('2')
    assert un.edges['2'] == list('1')


def test_add_edge_should_throw_an_error_if_from_node_does_not_exist():
    un = n.UndirectedNetwork()
    un.add_node(2)
    with pytest.raises(ValueError):
        un.add_edge(1, 2)


def test_add_edge_should_throw_an_error_if_to_node_does_not_exist():
    un = n.UndirectedNetwork()
    un.add_node(2)
    with pytest.raises(ValueError):
        un.add_edge(1, 2)


def test_add_edge_and_remove_edge_should_leave_network_in_unchanged_state():
    un = n.UndirectedNetwork()
    un.add_node(1)
    un.add_node(2)
    copied_dn = copy.deepcopy(un)
    un.add_edge(1, 2)
    un.remove_edge(1, 2)
    assert un == copied_dn


def test_create_nodes_and_edge_should_be_equal_to_adding_nodes_and_edge():
    un_one = n.UndirectedNetwork()
    un_one.add_node(1)
    un_one.add_node(2)
    un_one.add_edge(1, 2)
    un_two = n.UndirectedNetwork()
    un_two.create_nodes_and_edge(1, 2)
    assert un_one == un_two


def test_create_nodes_and_edge_should_be_idempotent():
    un = n.UndirectedNetwork()
    un.create_nodes_and_edge(1, 2)
    copied_un = copy.deepcopy(un)
    copied_un.create_nodes_and_edge(1, 2)
    assert un == copied_un


def test_add_node_unique_should_add_new_node_every_time():
    un = n.UndirectedNetwork()
    un.add_node_unique(1)
    un.add_node_unique(1)
    un.add_node_unique(1)
    assert len(un.nodes) == 3


def test_network_should_remember_order_of_insertion():
    un = n.UndirectedNetwork()
    un.add_node(3)
    un.add_node(2)
    un.add_node(5)
    assert list(un.nodes.keys()) == ['3', '2', '5']


def test_network_should_remember_order_of_insertion_even_when_element_removed():
    un = n.UndirectedNetwork()
    un.add_node(3)
    un.add_node(2)
    un.add_node(5)
    un.remove_node(2)
    assert list(un.nodes.keys()) == ['3', '5']


def test_undirected_network_should_throw_on_non_square_adj_matrix():
    with pytest.raises(ValueError):
        n.UndirectedNetwork(np.array([[1, 2, 3], [3, 2, 1]]))


def test_undirected_network_should_throw_on_non_even_entry_in_diagonal():
    with pytest.raises(ValueError):
        n.UndirectedNetwork(np.array([[1, 0], [0, 1]]))


def test_undirected_network_should_throw_on_non_symetric_adjacency_matrix():
    with pytest.raises(ValueError):
        n.UndirectedNetwork(np.array([[1, 0], [0, 0]]))


def test_should_create_undirected_network_from_single_entry():
    dn = n.UndirectedNetwork(np.array([0]))
    assert len(dn.nodes) == 1
    assert nodes_equal(dn.nodes['1'], tn.Node(0, name="1"))
    assert len(dn.edges) == 0


def test_should_be_able_to_create_undirected_network_from_adj_matrix():
    adj_matrix = np.array(
        [[0, 1, 0, 0, 3, 0],
         [1, 2, 2, 1, 0, 0],
         [0, 2, 0, 1, 1, 1],
         [0, 1, 1, 0, 0, 0],
         [3, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 2]])
    un_from_adj = n.UndirectedNetwork(adj_matrix)

    un = n.UndirectedNetwork()
    for node in range(1, 7):
        un.add_node(0, str(node))
    un.add_edge("1", "2")
    un.add_edge("1", "5")
    un.add_edge("1", "5")
    un.add_edge("1", "5")

    un.add_edge("2", "2")
    un.add_edge("2", "3")
    un.add_edge("2", "3")
    un.add_edge("2", "4")

    un.add_edge("3", "4")
    un.add_edge("3", "5")
    un.add_edge("3", "6")

    un.add_edge("6", "6")
    assert un_from_adj == un
    assert (adj_matrix == un.adjacency_matrix).all()