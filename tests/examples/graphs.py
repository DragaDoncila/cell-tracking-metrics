import networkx as nx

from traccuracy._tracking_graph import TrackingGraph
from traccuracy.matchers._base import Matched

"""A set of fixtures covering basic graph matching cases over 3 time frames
Covers edge cases, good matchings, fn nodes, fp nodes, two to one matchings in each
direction (pred -> gt, gt -> pred), and divisions
The type of mapping e.g. one to one or one to many is annotated in the docstring
Please note that the names of the fixtures are just meant to be descriptive and may
or may not match to a particular error type as described by a set of metrics

example of how to use
@pytest.mark.parametrize("i", [0, 1, 2], ids=["0", "1", "2"])
def test_fn_node(i):
    matched = fn_node_matched(i)
    assert ...
"""


def basic_graph(node_ids=(1, 2, 3), y_offset=0, frame_key="t", location_keys=("y")):
    nodes = []
    for t, node in enumerate(node_ids):
        nodes.append((node, {frame_key: t, location_keys[0]: 0 + y_offset}))

    edges = []
    for i in range(len(node_ids) - 1):
        edges.append((node_ids[i], node_ids[i + 1]))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return TrackingGraph(graph, frame_key=frame_key, location_keys=location_keys)


# edge cases
def empty_pred():
    gt = basic_graph()
    pred = TrackingGraph(nx.DiGraph())
    mapping = []
    return Matched(gt, pred, mapping, {})


def empty_gt():
    pred = basic_graph()
    gt = TrackingGraph(nx.DiGraph())
    mapping = []
    return Matched(gt, pred, mapping, {})


# good
def good_matched():
    """one to one"""
    gt = basic_graph()
    pred = basic_graph(node_ids=(4, 5, 6), y_offset=1)
    mapping = [(1, 4), (2, 5), (3, 6)]
    return Matched(gt, pred, mapping, {})


# fn_node
def fn_node_matched(time_to_drop):  # 0, 1, or 2
    """one to one"""
    gt = basic_graph()
    pred_node_ids = (4, 5, 6)
    pred = basic_graph(node_ids=pred_node_ids, y_offset=1)
    mapping = [(1, 4), (2, 5), (3, 6)]
    pred.graph.remove_node(pred_node_ids[time_to_drop])
    del mapping[time_to_drop]
    return Matched(gt, pred, mapping, {})


# fn_edge
def fn_edge_matched(edge_to_drop):  # 0 or 1
    """one to one"""
    gt = basic_graph()
    pred_node_ids = (4, 5, 6)
    pred = basic_graph(node_ids=pred_node_ids, y_offset=1)
    edge = (pred_node_ids[edge_to_drop], pred_node_ids[edge_to_drop + 1])
    pred.graph.remove_edge(*edge)
    mapping = [(1, 4), (2, 5), (3, 6)]
    return Matched(gt, pred, mapping, {})


# fp_node
def fp_node_matched(time_to_add):  # 0, 1, or 2
    """one to one"""
    gt = basic_graph()
    pred_node_ids = (4, 5, 6)
    pred = basic_graph(node_ids=pred_node_ids, y_offset=1)
    pred.graph.add_node(7, **{"t": time_to_add, "x": time_to_add, "y": 2})
    mapping = [(1, 4), (2, 5), (3, 6)]
    return Matched(gt, pred, mapping, {})


# fp_edge
def fp_edge_matched(edge_to_add):  # 0 or 1
    """one to one"""
    gt = basic_graph()
    pred_node_ids = (4, 5, 6)
    pred = basic_graph(node_ids=pred_node_ids, y_offset=1)
    pred.graph.add_node(7, **{"t": edge_to_add, "y": 2})
    pred.graph.add_node(8, **{"t": edge_to_add + 1, "y": 2})
    pred.graph.add_edge(7, 8)
    mapping = [(1, 4), (2, 5), (3, 6)]
    return Matched(gt, pred, mapping, {})


# identity switch
def crossover_edge():
    """one to one"""
    gta = basic_graph().graph
    gtb = basic_graph(node_ids=(4, 5, 6), y_offset=1).graph
    gta.add_nodes_from(gtb.nodes(data=True))
    gta.add_edges_from(gtb.edges(data=True))
    gt = TrackingGraph(gta, location_keys=("y"))
    pred = basic_graph(node_ids=(7, 8, 9), y_offset=0.5)
    mapping = [(4, 7), (2, 8), (3, 9)]
    return Matched(gt, pred, mapping, {})


# two pred to one gt (identity switch)
def node_one_to_two(time):  # 0, 1, or 2
    """one to many"""
    gt_node_ids = (1, 2, 3)
    gt = basic_graph(node_ids=gt_node_ids, y_offset=1)
    pred_node_ids = (4, 5, 6)
    pred = basic_graph(node_ids=pred_node_ids, y_offset=0)
    pred.graph.add_node(7, **{"t": time, "y": 2})
    mapping = [(1, 4), (2, 5), (3, 6)]
    if time == 1:
        pred.graph.remove_edge(5, 6)
        pred.graph.add_edge(7, 6)
        pred.graph.nodes[6]["y"] = 2
    mapping.append((gt_node_ids[time], 7))
    return Matched(gt, pred, mapping, {})


# two pred edges to one gt edge
def edge_one_to_two(time):  # 0 or 1
    gt = basic_graph()

    a = basic_graph(node_ids=(4, 5, 6), y_offset=1).graph
    b = basic_graph(node_ids=(7, 8, 9), y_offset=-1).graph
    a.add_nodes_from(b.nodes(data=True))
    a.add_edges_from(b.edges(data=True))
    pred = TrackingGraph(a, location_keys=("y"))

    mapping = [(1, 4), (2, 5), (3, 6)]
    if time == 0:
        mapping.extend([(1, 7), (2, 8)])
    elif time == 1:
        mapping.extend([(2, 8), (3, 9)])
    return Matched(gt, pred, mapping, {})


# two gt to one pred (non split vertex)
def node_two_to_one(time):  # 0, 1, or 2
    """many to one"""
    gt_node_ids = (1, 2, 3)
    gt = basic_graph(node_ids=gt_node_ids, y_offset=0)
    pred_node_ids = (4, 5, 6)
    pred = basic_graph(node_ids=pred_node_ids, y_offset=1)
    gt.graph.add_node(7, **{"t": time, "y": 2})
    mapping = [(1, 4), (2, 5), (3, 6)]
    if time == 1:
        gt.graph.remove_edge(1, 2)
        gt.graph.add_edge(1, 7)
        gt.graph.nodes[1]["y"] = 2
    mapping.append((7, pred_node_ids[time]))
    return Matched(gt, pred, mapping, {})


def edge_two_to_one(time):  # 0 or 1
    """many to one"""
    a = basic_graph(y_offset=1).graph
    b = basic_graph(node_ids=(4, 5, 6), y_offset=-1).graph
    a.add_nodes_from(b.nodes(data=True))
    a.add_edges_from(b.edges(data=True))
    gt = TrackingGraph(a, location_keys=("y"))

    pred = basic_graph(node_ids=(7, 8, 9))

    mapping = [(1, 7), (2, 8), (3, 9)]
    if time == 0:
        mapping.extend([(4, 7), (5, 8)])
    elif time == 1:
        mapping.extend([(5, 8), (6, 9)])
    return Matched(gt, pred, mapping, {})


def gap_close_gt_gap():
    gt = basic_graph(node_ids=(1, 2, 3, 4)).graph
    pred = basic_graph(node_ids=(5, 6, 7, 8), y_offset=-1)

    # Ablate node 2 from gt
    gt.remove_node(2)
    gt.add_edge(1, 3)
    gt = TrackingGraph(gt, location_keys=("y"))

    mapping = [(1, 5), (3, 7), (4, 8)]
    return Matched(gt, pred, mapping, {})


def gap_close_pred_gap():
    gt = basic_graph(node_ids=(1, 2, 3, 4))
    pred = basic_graph(node_ids=(5, 6, 7, 8), y_offset=-1).graph

    # Remove pred node 7 to create gap
    pred.remove_node(7)
    pred.add_edge(6, 8)
    pred = TrackingGraph(pred, location_keys=("y"))

    mapping = [(1, 5), (2, 6), (4, 8)]
    return Matched(gt, pred, mapping, {})


def gap_close_matched_gap():
    gt = basic_graph(node_ids=(1, 2, 3, 4)).graph
    pred = basic_graph(node_ids=(5, 6, 7, 8), y_offset=-1).graph

    gt.remove_node(2)
    gt.add_edge(1, 3)
    pred.remove_node(6)
    pred.add_edge(5, 7)

    mapping = [(1, 5), (3, 7), (4, 8)]
    return Matched(
        TrackingGraph(gt, location_keys=("y")),
        TrackingGraph(pred, location_keys=("y")),
        mapping,
        {},
    )


def gap_close_offset():
    gt = basic_graph(node_ids=(1, 2, 3, 4)).graph
    pred = basic_graph(node_ids=(5, 6, 7, 8), y_offset=-1).graph

    gt.remove_node(2)
    gt.add_edge(1, 3)
    pred.remove_node(7)
    pred.add_edge(6, 8)

    mapping = [(1, 5), (4, 8)]
    return Matched(
        TrackingGraph(gt, location_keys=("y")),
        TrackingGraph(pred, location_keys=("y")),
        mapping,
        {},
    )


def get_division_graphs():
    """
    G1
                                2_4
    1_0 -- 1_1 -- 1_2 -- 1_3 -<
                                3_4
    G2
                  2_2 -- 2_3 -- 2_4
    1_0 -- 1_1 -<
                  3_2 -- 3_3 -- 3_4
    """

    G1 = nx.DiGraph()
    G1.add_edge("1_0", "1_1")
    G1.add_edge("1_1", "1_2")
    G1.add_edge("1_2", "1_3")
    G1.add_edge("1_3", "2_4")
    G1.add_edge("1_3", "3_4")

    attrs = {}
    for node in G1.nodes:
        attrs[node] = {"t": int(node[-1:]), "x": 0, "y": 0}
    nx.set_node_attributes(G1, attrs)

    G2 = nx.DiGraph()
    G2.add_edge("1_0", "1_1")
    # Divide to generate 2 lineage
    G2.add_edge("1_1", "2_2")
    G2.add_edge("2_2", "2_3")
    G2.add_edge("2_3", "2_4")
    # Divide to generate 3 lineage
    G2.add_edge("1_1", "3_2")
    G2.add_edge("3_2", "3_3")
    G2.add_edge("3_3", "3_4")

    attrs = {}
    for node in G2.nodes:
        attrs[node] = {"t": int(node[-1:]), "x": 0, "y": 0}
    nx.set_node_attributes(G2, attrs)

    mapper = [("1_0", "1_0"), ("1_1", "1_1"), ("2_4", "2_4"), ("3_4", "3_4")]

    return G1, G2, mapper


def basic_division_t0(start_id=1, y_offset=0, frame_key="t", location_keys=("y")):
    nodes = [
        (start_id, {frame_key: 0, location_keys[0]: y_offset}),
        (start_id + 1, {frame_key: 1, location_keys[0]: y_offset + 0.5}),
        (start_id + 2, {frame_key: 1, location_keys[0]: y_offset - 0.5}),
        (start_id + 3, {frame_key: 2, location_keys[0]: y_offset + 0.5}),
        (start_id + 4, {frame_key: 2, location_keys[0]: y_offset - 0.5}),
    ]
    edges = [
        (start_id, start_id + 1),
        (start_id, start_id + 2),
        (start_id + 1, start_id + 3),
        (start_id + 2, start_id + 4),
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return TrackingGraph(graph, frame_key=frame_key, location_keys=location_keys)


def basic_division_t1(start_id=1, y_offset=0, frame_key="t", location_keys=("y")):
    nodes = [
        (start_id, {frame_key: 0, location_keys[0]: y_offset}),
        (start_id + 1, {frame_key: 1, location_keys[0]: y_offset}),
        (start_id + 2, {frame_key: 2, location_keys[0]: y_offset + 0.5}),
        (start_id + 3, {frame_key: 2, location_keys[0]: y_offset - 0.5}),
    ]
    edges = [
        (start_id, start_id + 1),
        (start_id + 1, start_id + 2),
        (start_id + 1, start_id + 3),
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return TrackingGraph(graph, frame_key=frame_key, location_keys=location_keys)


def basic_division_t2(start_id=1, y_offset=0, frame_key="t", location_keys=("y")):
    nodes = [
        (start_id, {frame_key: 0, location_keys[0]: y_offset}),
        (start_id + 1, {frame_key: 1, location_keys[0]: y_offset}),
        (start_id + 2, {frame_key: 2, location_keys[0]: y_offset}),
        (start_id + 3, {frame_key: 3, location_keys[0]: y_offset + 0.5}),
        (start_id + 4, {frame_key: 3, location_keys[0]: y_offset - 0.5}),
    ]
    edges = [
        (start_id, start_id + 1),
        (start_id + 1, start_id + 2),
        (start_id + 2, start_id + 3),
        (start_id + 2, start_id + 4),
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return TrackingGraph(graph, frame_key=frame_key, location_keys=location_keys)


def basic_division(t_div, start_id=1, y_offset=0, frame_key="t", location_keys=("y")):
    """one to one"""
    if t_div == 0:
        return basic_division_t0(start_id, y_offset, frame_key, location_keys)
    elif t_div == 1:
        return basic_division_t1(start_id, y_offset, frame_key, location_keys)
    elif t_div == 2:
        return basic_division_t2(start_id, y_offset, frame_key, location_keys)


def longer_division(t_div, start_id=1, y_offset=0, frame_key="t", location_keys=("y")):
    """one to one"""
    nodes = []
    nid = start_id
    for t in range(5):
        if t <= t_div:
            nodes.append((nid, {frame_key: t, location_keys[0]: y_offset}))
            nid += 1
        else:
            nodes.extend(
                [
                    (nid, {frame_key: t, location_keys[0]: y_offset + 0.5}),
                    (nid + 1, {frame_key: t, location_keys[0]: y_offset - 0.5}),
                ]
            )
            nid += 2
    edges = []
    for t in range(4):
        if t < t_div:
            edges.append((start_id + t, start_id + t + 1))
        elif t == t_div:
            edges.extend(
                [(start_id + t, start_id + t + 1), (start_id + t, start_id + t + 2)]
            )
        else:
            delta = start_id + t_div + 2 * (t - t_div) - 1
            edges.extend([(delta, delta + 2), (delta + 1, delta + 3)])

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return TrackingGraph(graph, frame_key=frame_key, location_keys=location_keys)


def empty_pred_div(t_div):
    gt = basic_division(t_div)
    pred = TrackingGraph(nx.DiGraph())
    mapping = []
    return Matched(gt, pred, mapping, {})


def empty_gt_div(t_div):
    gt = TrackingGraph(nx.DiGraph())
    pred = basic_division(t_div)
    mapping = []
    return Matched(gt, pred, mapping, {})


def good_div(t_div):
    """one to one"""
    gt = basic_division(t_div)
    start_id = max(gt.graph.nodes) + 1
    pred = basic_division(t_div, start_id=start_id, y_offset=0.5)
    mapping = list(zip(range(1, start_id), range(start_id, start_id * 2)))
    return Matched(gt, pred, mapping, {})


def fp_div(t_div):
    """one to one"""
    # t_div either 0 or 1
    gt = basic_division(t_div)
    start_id = max(gt.graph.nodes) + 1
    pred = basic_division(t_div, start_id=start_id, y_offset=0.5)
    mapping = list(zip(range(1, start_id), range(start_id, start_id * 2)))
    if t_div == 0:
        gt.graph.remove_edge(1, 2)
    elif t_div == 1:
        gt.graph.remove_edge(2, 3)
    return Matched(gt, pred, mapping, {})


def one_child(t_div):
    """one to one"""
    # t_div either 0 or 1
    gt = basic_division(t_div)
    start_id = max(gt.graph.nodes) + 1
    pred = basic_division(t_div, start_id=start_id, y_offset=0.5)
    mapping = list(zip(range(1, start_id), range(start_id, start_id * 2)))
    pred.graph.remove_edge(6, 7)
    return Matched(gt, pred, mapping, {})


def no_children(t_div):
    """one to one"""
    # t_div either 0 or 1
    gt = basic_division(t_div)
    start_id = max(gt.graph.nodes) + 1
    pred = basic_division(t_div, start_id=start_id, y_offset=0.5)
    mapping = list(zip(range(1, start_id), range(start_id, start_id * 2)))
    pred.graph.remove_edge(6, 7)
    pred.graph.remove_edge(6, 8)
    return Matched(gt, pred, mapping, {})


def wrong_child(t_div):
    """one to one"""
    # t_div either 0 or 1
    gt_graph = basic_division(t_div).graph
    child_start_id = max(gt_graph.nodes)
    # Add additional gt node and/or edge for wrong child
    nodes = [
        (child_start_id + 1, {"t": t_div + 1, "y": -0.25}),
        (child_start_id + 2, {"t": t_div + 2, "y": -0.25}),
    ]
    if t_div == 0:
        gt_graph.add_nodes_from(nodes)
        gt_graph.add_edge(child_start_id + 1, child_start_id + 2)
    elif t_div == 1:
        gt_graph.add_nodes_from(nodes[0:1])
    gt = TrackingGraph(gt_graph, frame_key="t", location_keys=("y"))
    start_id = max(gt.graph.nodes) + 1
    pred = basic_division(t_div, start_id=start_id, y_offset=0.5)

    # mapping of the two basic div graphs
    mapping = list(zip(range(1, child_start_id + 1), range(start_id, start_id * 2)))
    # remove mapping to one of the correct daughters and add to wrong daughter
    if t_div == 0:
        mapping.remove((3, 10))
        mapping.remove((5, 12))
        mapping.extend([(6, 10), (7, 12)])
    elif t_div == 1:
        mapping.remove((4, 9))
        mapping.append((5, 9))
    return Matched(gt, pred, mapping, {})


def wrong_children(t_div):
    """one to one"""
    # t_div either 0 or 1
    gt_graph = basic_division(t_div).graph
    gt = TrackingGraph(gt_graph, frame_key="t", location_keys=("y"))
    start_id = max(gt.graph.nodes) + 1
    pred = basic_division(t_div, start_id=start_id, y_offset=0.5)

    # mapping of the two basic div graphs
    mapping = list(zip(range(1, start_id), range(start_id, start_id * 2)))
    # remove mapping between daughters
    if t_div == 0:
        mapping.remove((3, 8))
        mapping.remove((2, 7))
    elif t_div == 1:
        mapping.remove((4, 8))
        mapping.remove((3, 7))
    return Matched(gt, pred, mapping, {})


def div_1early_end():
    """one to one"""
    gt = longer_division(1)
    start_id = max(gt.nodes) + 1
    pred = longer_division(0, start_id=start_id, y_offset=0.75)
    mapping = [(1, 9), (2, 11), (4, 13), (6, 15), (8, 17), (3, 12), (5, 14), (7, 16)]
    return Matched(gt, pred, mapping, {})


def div_1early_mid():
    """one to one"""
    gt = longer_division(2)
    start_id = max(gt.nodes) + 1
    pred = longer_division(1, start_id=start_id, y_offset=0.75)
    mapping = [(1, 8), (2, 9), (3, 11), (5, 13), (7, 15), (4, 12), (6, 14)]
    return Matched(gt, pred, mapping, {})


def div_2early_end():
    """one to one"""
    gt = longer_division(2)
    start_id = max(gt.nodes) + 1
    pred = longer_division(0, start_id=start_id, y_offset=0.75)
    mapping = [(1, 8), (2, 10), (3, 12), (5, 14), (7, 16), (4, 13), (6, 15)]
    return Matched(gt, pred, mapping, {})


def div_2early_mid():
    """one to one"""
    gt = longer_division(3)
    start_id = max(gt.nodes) + 1
    pred = longer_division(1, start_id=start_id, y_offset=0.75)
    mapping = [(1, 7), (2, 8), (3, 10), (4, 12), (6, 14), (5, 13)]
    return Matched(gt, pred, mapping, {})


def div_1late_end():
    """one to one"""
    gt = longer_division(0)
    start_id = max(gt.nodes) + 1
    pred = longer_division(1, start_id=start_id, y_offset=0.75)
    mapping = [(1, 10), (2, 11), (4, 12), (6, 14), (8, 16), (5, 13), (7, 15), (9, 17)]
    return Matched(gt, pred, mapping, {})


def div_1late_mid():
    """one to one"""
    gt = longer_division(1)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id=start_id, y_offset=0.75)
    mapping = [(1, 9), (2, 10), (3, 11), (6, 13), (8, 15), (5, 12), (7, 14)]
    return Matched(gt, pred, mapping, {})


def div_2late_end():
    """one to one"""
    gt = longer_division(0)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id=start_id, y_offset=0.75)
    mapping = [(1, 10), (2, 11), (4, 12), (6, 13), (8, 15), (7, 14), (9, 16)]
    return Matched(gt, pred, mapping, {})


def div_2late_mid():
    """one to one"""
    gt = longer_division(1)
    start_id = max(gt.nodes) + 1
    pred = longer_division(3, start_id=start_id, y_offset=0.75)
    mapping = [(1, 9), (2, 10), (3, 11), (5, 12), (7, 13), (8, 14)]
    return Matched(gt, pred, mapping, {})


def div_shift_min_match():
    """one to one
    Minimal requirements for matching in a shifted division
    - Early division node is matched to a predecessor of the late division
    - Daughters must be matched in the frame following the late division
    """
    gt = longer_division(1)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id=start_id, y_offset=0.75)
    mapping = [(2, 10), (6, 13), (5, 12)]
    return Matched(gt, pred, mapping, {})


def div_shift_bad_match_pred():
    """one to one
    Division isn't corrected because the early division node isn't matched
    to a predecessor of the late division node
    """
    gt = longer_division(1)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id=start_id, y_offset=0.75)
    mapping = [(1, 9), (6, 13), (5, 12)]
    return Matched(gt, pred, mapping, {})


def div_shift_bad_match_daughter():
    """one to one
    Division isn't corrected because daughter nodes aren't matched in frame
    following late division
    """
    gt = longer_division(1)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id=start_id, y_offset=0.75)
    mapping = [(2, 10), (7, 14), (8, 15)]
    return Matched(gt, pred, mapping, {})


def div_parent_gap():
    gt = longer_division(2)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id, y_offset=0.75).graph

    # Remove parent node from prediction
    pred.remove_node(10)
    # Add gap closing edges to daughters
    pred.add_edges_from([(9, 11), (9, 12)])
    pred = TrackingGraph(pred, location_keys=("y"))

    mapping = [(1, 8), (2, 9), (4, 11), (5, 12), (6, 13), (7, 14)]
    return Matched(gt, pred, mapping, {})


def div_daughter_gap():
    gt = longer_division(2)
    start_id = max(gt.nodes) + 1
    pred = longer_division(2, start_id, y_offset=0.75).graph

    # Remove one daughter node from prediction
    pred.remove_node(11)
    pred.add_edge(10, 13)
    pred = TrackingGraph(pred, location_keys=("y"))

    mapping = [(1, 8), (2, 9), (3, 10), (5, 12), (6, 13), (7, 14)]
    return Matched(gt, pred, mapping, {})
