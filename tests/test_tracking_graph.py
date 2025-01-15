from collections import Counter

import networkx as nx
import pytest

from traccuracy import EdgeFlag, NodeFlag, TrackingGraph


@pytest.fixture
def nx_comp1():
    """Component 1: Y=1
    x
    3|
    2|       /--1_3--1_4
    1| 1_0--1_1
    0|       \\--1_2
    ---------------------- t
        0    1   2    3
    """
    cells = [
        {"id": "1_0", "t": 0, "y": 1, "x": 1},
        {"id": "1_1", "t": 1, "y": 1, "x": 1, "is_tp_division": True},
        {"id": "1_2", "t": 2, "y": 1, "x": 0},
        {"id": "1_3", "t": 2, "y": 1, "x": 2},
        {"id": "1_4", "t": 3, "y": 1, "x": 2},
    ]

    edges = [
        {"source": "1_0", "target": "1_1", "is_tp": True},
        {"source": "1_1", "target": "1_2", "is_tp": False},
        {"source": "1_1", "target": "1_3"},
        {"source": "1_3", "target": "1_4"},
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from([(cell["id"], cell) for cell in cells])
    graph.add_edges_from([(edge["source"], edge["target"], edge) for edge in edges])
    return graph


@pytest.fixture
def nx_comp2():
    """Component 2: X=1
    y
    3|              /--2_4
    2|  2_0--2_1--2_2
    1|             \\--2_3
    0|
    ---------------------- t
        0    1     2    3
    """
    cells = [
        {"id": "2_0", "t": 0, "y": 2, "x": 1},
        {"id": "2_1", "t": 1, "y": 2, "x": 1},
        {"id": "2_2", "t": 2, "y": 2, "x": 1, "is_tp_division": True},
        {"id": "2_3", "t": 3, "y": 1, "x": 1},
        {"id": "2_4", "t": 3, "y": 3, "x": 1},
    ]

    edges = [
        {"source": "2_0", "target": "2_1"},
        {"source": "2_1", "target": "2_2"},
        {"source": "2_2", "target": "2_3"},
        {"source": "2_2", "target": "2_4"},
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from([(cell["id"], cell) for cell in cells])
    graph.add_edges_from([(edge["source"], edge["target"]) for edge in edges])
    return graph


@pytest.fixture
def nx_merge():
    """
    3_0--3_1--\\
              3_2--3_3
    3_4--3_5--/
    """
    cells = [
        {"id": "3_0", "t": 0, "x": 0, "y": 0},
        {"id": "3_1", "t": 1, "x": 0, "y": 0},
        {"id": "3_2", "t": 2, "x": 0, "y": 0},
        {"id": "3_3", "t": 3, "x": 0, "y": 0},
        {"id": "3_4", "t": 0, "x": 0, "y": 0},
        {"id": "3_5", "t": 1, "x": 0, "y": 0},
    ]

    edges = [
        {"source": "3_0", "target": "3_1"},
        {"source": "3_1", "target": "3_2"},
        {"source": "3_2", "target": "3_3"},
        {"source": "3_4", "target": "3_5"},
        {"source": "3_5", "target": "3_2"},
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from([(cell["id"], cell) for cell in cells])
    graph.add_edges_from([(edge["source"], edge["target"]) for edge in edges])
    return graph


@pytest.fixture
def merge_graph(nx_merge):
    return TrackingGraph(nx_merge)


@pytest.fixture
def simple_graph(nx_comp1):
    return TrackingGraph(nx_comp1)


@pytest.fixture
def complex_graph(nx_comp1, nx_comp2):
    return TrackingGraph(nx.compose(nx_comp1, nx_comp2))


def test_constructor(nx_comp1):
    tracking_graph = TrackingGraph(nx_comp1)
    assert tracking_graph.start_frame == 0
    assert tracking_graph.end_frame == 4
    assert tracking_graph.nodes_by_frame == {
        0: {"1_0"},
        1: {"1_1"},
        2: {"1_2", "1_3"},
        3: {"1_4"},
    }

    # raise AssertionError if frame key not present or ValueError if overlaps
    # with reserved values
    with pytest.raises(AssertionError, match=r"Frame key .* not present for node .*."):
        TrackingGraph(nx_comp1, frame_key="f")
    with pytest.raises(ValueError):
        TrackingGraph(nx_comp1, frame_key=NodeFlag.CTC_FALSE_NEG)
    with pytest.raises(AssertionError):
        TrackingGraph(nx_comp1, location_keys=["x", "y", "z"])
    with pytest.raises(ValueError):
        TrackingGraph(nx_comp1, location_keys=["x", NodeFlag.CTC_FALSE_NEG])


def test_get_cells_by_frame(simple_graph):
    assert Counter(simple_graph.nodes_by_frame[0]) == Counter({"1_0"})
    assert Counter(simple_graph.nodes_by_frame[2]) == Counter(["1_2", "1_3"])
    # Test non-existent frame
    assert Counter(simple_graph.nodes_by_frame[5]) == Counter([])


def test_get_nodes_with_flag(simple_graph):
    assert Counter(simple_graph.get_nodes_with_flag(NodeFlag.TP_DIV)) == Counter(
        ["1_1"]
    )
    assert Counter(simple_graph.get_nodes_with_flag(NodeFlag.FP_DIV)) == Counter([])
    with pytest.raises(ValueError):
        assert simple_graph.get_nodes_with_flag("is_tp_division")


def test_get_edges_with_flag(simple_graph):
    assert Counter(simple_graph.get_edges_with_flag(EdgeFlag.TRUE_POS)) == Counter(
        [("1_0", "1_1")]
    )
    assert Counter(simple_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_NEG)) == Counter(
        []
    )
    with pytest.raises(ValueError):
        assert simple_graph.get_nodes_with_flag("is_tp")


def test_get_divisions(complex_graph):
    assert complex_graph.get_divisions() == ["1_1", "2_2"]


def test_get_merges(merge_graph):
    assert merge_graph.get_merges() == ["3_2"]


def test_get_connected_components(complex_graph, nx_comp1, nx_comp2):
    tracks = complex_graph.get_connected_components()
    assert len(tracks) == 2
    if "1_1" in tracks[0].graph:
        track1 = tracks[0]
        track2 = tracks[1]
    else:
        track1 = tracks[1]
        track2 = tracks[0]
    assert track1.graph.nodes == nx_comp1.nodes
    assert track1.graph.edges == nx_comp1.edges
    assert track2.graph.nodes == nx_comp2.nodes
    assert track2.graph.edges == nx_comp2.edges


def test_get_subgraph(simple_graph):
    target_nodes = ("1_0", "1_1")
    subgraph = simple_graph.get_subgraph(target_nodes)
    assert len(subgraph.nodes) == 2
    assert len(subgraph.edges) == 1
    # test that nodes_by_flag dicts are maintained
    assert Counter(subgraph.nodes_by_flag[NodeFlag.TP_DIV]) == Counter(["1_1"])
    assert Counter(subgraph.edges_by_flag[EdgeFlag.TRUE_POS]) == Counter(
        [("1_0", "1_1")]
    )
    # test that start and end frame are updated
    assert subgraph.start_frame == 0
    assert subgraph.end_frame == 2

    # test empty target nodes
    empty_graph = simple_graph.get_subgraph([])
    assert Counter(empty_graph.nodes) == Counter([])


def test_set_flag_on_node(simple_graph):
    assert simple_graph.nodes()["1_0"] == {"id": "1_0", "t": 0, "y": 1, "x": 1}
    assert simple_graph.nodes()["1_1"] == {
        "id": "1_1",
        "t": 1,
        "y": 1,
        "x": 1,
        "is_tp_division": True,
    }

    simple_graph.set_flag_on_node("1_0", NodeFlag.CTC_FALSE_POS, value=True)
    assert simple_graph.nodes()["1_0"] == {
        "id": "1_0",
        "t": 0,
        "y": 1,
        "x": 1,
        NodeFlag.CTC_FALSE_POS: True,
    }
    assert "1_0" in simple_graph.nodes_by_flag[NodeFlag.CTC_FALSE_POS]

    simple_graph.set_flag_on_node("1_0", NodeFlag.CTC_FALSE_POS, value=False)
    assert simple_graph.nodes()["1_0"] == {
        "id": "1_0",
        "t": 0,
        "y": 1,
        "x": 1,
        NodeFlag.CTC_FALSE_POS: False,
    }
    assert "1_0" not in simple_graph.nodes_by_flag[NodeFlag.CTC_FALSE_POS]

    simple_graph.set_flag_on_all_nodes(NodeFlag.CTC_FALSE_POS, value=True)
    for node in simple_graph.nodes:
        assert simple_graph.nodes[node][NodeFlag.CTC_FALSE_POS] is True
    assert Counter(simple_graph.nodes_by_flag[NodeFlag.CTC_FALSE_POS]) == Counter(
        list(simple_graph.nodes())
    )

    simple_graph.set_flag_on_all_nodes(NodeFlag.CTC_FALSE_POS, value=False)
    for node in simple_graph.nodes:
        assert simple_graph.nodes[node][NodeFlag.CTC_FALSE_POS] is False
    assert not simple_graph.nodes_by_flag[NodeFlag.CTC_FALSE_POS]

    with pytest.raises(ValueError):
        simple_graph.set_flag_on_node("1_0", "x", 2)


def test_remove_flag_from_node(simple_graph):
    flag = NodeFlag.CTC_FALSE_POS
    simple_graph.set_flag_on_all_nodes(flag)

    simple_graph.remove_flag_from_node("1_3", flag)
    assert flag not in simple_graph.graph.nodes["1_3"]
    assert flag not in simple_graph.nodes_by_flag[flag]

    # Check that other nodes were unaffected
    for node in ["1_0", "1_1", "1_2", "1_4"]:
        assert flag in simple_graph.graph.nodes[node]
        assert node in simple_graph.nodes_by_flag[flag]

    # Error if flag not present
    with pytest.raises(KeyError, match=r".* not present on node .*"):
        simple_graph.remove_flag_from_node("1_3", NodeFlag.CTC_TRUE_POS)


def test_set_flag_on_edge(simple_graph):
    edge_id = ("1_1", "1_3")
    assert EdgeFlag.TRUE_POS not in simple_graph.edges()[edge_id]

    simple_graph.set_flag_on_edge(edge_id, EdgeFlag.TRUE_POS, value=True)
    assert simple_graph.edges()[edge_id][EdgeFlag.TRUE_POS] is True
    assert edge_id in simple_graph.edges_by_flag[EdgeFlag.TRUE_POS]

    simple_graph.set_flag_on_edge(edge_id, EdgeFlag.TRUE_POS, value=False)
    assert simple_graph.edges()[edge_id][EdgeFlag.TRUE_POS] is False
    assert edge_id not in simple_graph.edges_by_flag[EdgeFlag.TRUE_POS]

    simple_graph.set_flag_on_all_edges(EdgeFlag.CTC_FALSE_POS, value=True)
    for edge in simple_graph.edges:
        assert simple_graph.edges[edge][EdgeFlag.CTC_FALSE_POS] is True
    assert Counter(simple_graph.edges_by_flag[EdgeFlag.CTC_FALSE_POS]) == Counter(
        list(simple_graph.edges)
    )

    simple_graph.set_flag_on_all_edges(EdgeFlag.CTC_FALSE_POS, value=False)
    for edge in simple_graph.edges:
        assert simple_graph.edges[edge][EdgeFlag.CTC_FALSE_POS] is False
    assert not simple_graph.edges_by_flag[EdgeFlag.CTC_FALSE_POS]

    with pytest.raises(ValueError):
        simple_graph.set_flag_on_edge(("1_1", "1_3"), "x", 2)


def test_remove_flag_from_edge(simple_graph):
    flag = EdgeFlag.CTC_FALSE_POS
    simple_graph.set_flag_on_all_edges(flag)

    # Check basic removal
    edge = ("1_1", "1_3")
    simple_graph.remove_flag_from_edge(edge, flag)
    assert flag not in simple_graph.graph.edges[edge]
    assert edge not in simple_graph.edges_by_flag[flag]

    # Check other edges uneffected
    for edge in [("1_0", "1_1"), ("1_1", "1_2"), ("1_3", "1_4")]:
        assert flag in simple_graph.graph.edges[edge]
        assert edge in simple_graph.edges_by_flag[flag]

    # Error if flag not present
    with pytest.raises(KeyError, match=r".* not present on edge .*"):
        simple_graph.remove_flag_from_edge(("1_0", "1_1"), EdgeFlag.INTERTRACK_EDGE)


def test_get_tracklets(simple_graph):
    tracklets = simple_graph.get_tracklets()
    for tracklet in tracklets:
        start_nodes = [n for n, d in tracklet.graph.in_degree() if d == 0]
        assert len(start_nodes) == 1
        end_nodes = [n for n, d in tracklet.graph.out_degree() if d == 0]
        assert len(end_nodes)

        if start_nodes[0] == "1_0":
            assert end_nodes[0] == "1_1"
        elif start_nodes[0] == "1_2":
            assert end_nodes[0] == "1_2"
        elif start_nodes[0] == "1_3":
            assert end_nodes[0] == "1_4"
