import copy

import pytest
import tensornetwork as tn

import network_science.ns.network as n
from network_science.ns import compare_nodes


def test_directed_add_node_should_add_node():
    dn = n.DirectedNetwork()
    dn.add_node(1)
    assert len(dn.nodes) == 1
    assert compare_nodes(dn.nodes['1'], tn.Node(1, name="1"))


def test_directed_add_node_should_throw_if_node_already_exist():
    dn = n.DirectedNetwork()
    dn.add_node(1)
    with pytest.raises(ValueError):
        dn.add_node(1)


def test_adding_and_removing_node_should_leave_network_in_unchanged_state():
    dn = n.DirectedNetwork()
    dn.add_node(1)
    copied_dn = copy.deepcopy(dn)
    dn.add_node(2)
    removed = dn.remove_node(2)
    assert removed is True
    assert dn == copied_dn


def test_removing_non_existing_node_should_throw_an_error():
    dn = n.DirectedNetwork()
    assert dn.remove_node(2) is False


def test_add_edge_should_connect_two_nodes():
    dn = n.DirectedNetwork()
    dn.add_node(1)
    dn.add_node(2)
    dn.add_edge(1, 2)
    assert dn.edges['1'] == list('2')


def test_add_edge_should_throw_an_error_if_from_node_does_not_exist():
    dn = n.DirectedNetwork()
    dn.add_node(2)
    with pytest.raises(ValueError):
        dn.add_edge(1, 2)


def test_add_edge_should_throw_an_error_if_to_node_does_not_exist():
    dn = n.DirectedNetwork()
    dn.add_node(2)
    with pytest.raises(ValueError):
        dn.add_edge(1, 2)


def test_add_edge_and_remove_edge_should_leave_network_in_unchanged_state():
    dn = n.DirectedNetwork()
    dn.add_node(1)
    dn.add_node(2)
    copied_dn = copy.deepcopy(dn)
    dn.add_edge(1, 2)
    dn.remove_edge(1, 2)
    assert dn == copied_dn


def test_create_nodes_and_edge_should_be_equal_to_adding_nodes_and_edge():
    dn_one = n.DirectedNetwork()
    dn_one.add_node(1)
    dn_one.add_node(2)
    dn_one.add_edge(1, 2)
    dn_two = n.DirectedNetwork()
    dn_two.create_nodes_and_edge(1, 2)
    assert dn_one == dn_two


def test_create_nodes_and_edge_should_be_idempotent():
    dn = n.DirectedNetwork()
    dn.create_nodes_and_edge(1, 2)
    copied_dn = copy.deepcopy(dn)
    copied_dn.create_nodes_and_edge(1, 2)
    assert dn == copied_dn


def test_add_node_unique_should_add_new_node_every_time():
    dn = n.DirectedNetwork()
    dn.add_node_unique(1)
    dn.add_node_unique(1)
    dn.add_node_unique(1)
    assert len(dn.nodes) == 3
