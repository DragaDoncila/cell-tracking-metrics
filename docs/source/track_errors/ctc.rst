CTC Errors
==========

These node and edge error annotations are used to calculate the CTC metrics TRA and DET as well as the basic AOGM metric as described in :doc:`../metrics/ctc`.

.. jupyter-execute::
    :hide-code:
    
    import sys
    sys.path.append('../../tests')

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from traccuracy._tracking_graph import TrackingGraph
    from traccuracy.matchers import Matched

    import examples.graphs as ex_graphs
    import examples.segs as ex_segs

    def get_loc(graph, node):
        return graph.graph.nodes[node]["t"], graph.graph.nodes[node]["y"]


    def plot_graph(ax, graph: TrackingGraph, color="black", annotations={}, ann_color="red"):
        if graph.graph.number_of_nodes() == 0:
            return [0, 0], [0, 0]
        ids = list(graph.graph.nodes)
        x = [graph.graph.nodes[node]["t"] for node in ids]
        y = [graph.graph.nodes[node]["y"] for node in ids]
        ax.scatter(x, y, color=color)
        for _x, _y, _id in zip(x, y, ids):
            ax.text(_x + 0.05, _y + 0.05, str(_id))
            # Plot annotation if available
            ann = annotations.get(_id)
            if ann:
                ax.text(_x + 0.05, _y - 0.25, ann, color='purple')

        for u, v in graph.graph.edges():
            xs = [graph.graph.nodes[u]["t"], graph.graph.nodes[v]["t"]]
            ys = [graph.graph.nodes[u]["y"], graph.graph.nodes[v]["y"]]
            ax.plot(xs, ys, color=color)
            # Plot edge annotation if available
            ann = annotations.get((u, v))
            if ann:
                xx = sum(xs) / 2
                yy = sum(ys) / 2
                ax.text(xx + 0.1, yy, ann, color='orange', horizontalalignment='center')

        return [max(x), min(x)], [max(y), min(y)]


    def plot_matching(ax, matched, color="grey"):
        for u, v in matched.mapping:
            xs = [
                matched.gt_graph.graph.nodes[u]["t"],
                matched.pred_graph.graph.nodes[v]["t"],
            ]
            ys = [
                matched.gt_graph.graph.nodes[u]["y"],
                matched.pred_graph.graph.nodes[v]["y"],
            ]
            ax.plot(xs, ys, color=color, linestyle="dashed")


    def plot_matched(examples, annotations, title):
        gt_color = "black"
        pred_color = "blue"
        mapping_color = "grey"
        fig, ax = plt.subplots(1, len(examples) + 1, figsize=(3 * len(examples) + 1, 2))
        for i, (matched, anns) in enumerate(zip(examples, annotations)):
            axis = ax[i]
            xbounds, ybounds = plot_graph(axis, matched.gt_graph, color=gt_color, annotations=anns)
            bounds = plot_graph(axis, matched.pred_graph, color=pred_color, annotations=anns)
            xbounds.extend(bounds[0])
            ybounds.extend(bounds[1])
            plot_matching(axis, matched, color=mapping_color)
            axis.set_ybound(min(ybounds) - 0.5, max(ybounds) + 0.5)
            axis.set_xbound(min(xbounds) - 0.5, max(xbounds) + 0.5)
            axis.set_ylabel("Y Value")
            axis.set_xlabel("Time")

        handles = [
            Patch(color=gt_color),
            Patch(color=pred_color),
            Patch(color=mapping_color),
            Patch(color='orange'),
            Patch(color='purple')
        ]
        labels = ["Ground Truth", "Prediction", "Mapping", "Edge Annotations", "Node Annotations"]
        ax[-1].legend(handles=handles, labels=labels, loc="center")
        ax[-1].set_axis_off()
        plt.tight_layout()
        fig.suptitle(title, y=1.05)

Nodes
-----

True Positives
^^^^^^^^^^^^^^

A true positive node is one that is matched to only one node in the predicted graph. Additionally, the predicted node is not matched to any other node in the ground truth. True positive nodes are annotated on both the ground truth and the predicted graph.

False Positives
^^^^^^^^^^^^^^^

False positive nodes are annotated on the predicted graph and correspond to a predicted node without a match in the ground truth graph.

False Negatives
^^^^^^^^^^^^^^^

False negative nodes are annotated on the ground truth graph and correspond to a ground truth node without a match in the predicted graph.

Non-Split
^^^^^^^^^

Non-split nodes are annotated on the predicted graph and correspond to a node in the prediction that has been matched to more than one node in the ground truth graph.

.. jupyter-execute::
    :hide-code:

    plot_matched([ex_graphs.node_two_to_one(t) for t in [0, 1]], [{4: "NS"}, {5: "NS"}], "Non-Split Nodes")


Edges
-----

False Positives
^^^^^^^^^^^^^^^

False positive edges are annotated on the predicted graph. An edge is considered a false positive if both nodes are true positive nodes, but the edge does not match to any edge in the ground truth graph. In the example below, edge (7, 8) is a false positive.

.. jupyter-execute::
    :hide-code:

    plot_matched([ex_graphs.crossover_edge()], [{(7,8): "FP"}], "")

False Negatives
^^^^^^^^^^^^^^^

False negative edges are annotated on the ground truth graph. An edge is considered a false negative if:

1. Either node is annotated as a false negative node

.. jupyter-execute::
    :hide-code:

    plot_matched([ex_graphs.fn_node_matched(0)], [{1: "FN", (1, 2): "FN"}], "")

2. The corresponding edge in the predicted graph does not exist between two true positive nodes

.. jupyter-execute::
    :hide-code:

    plot_matched([ex_graphs.fn_edge_matched(0)], [{(1, 2): "FN"}], "")

3. Either node matches to a non-split node in the predicted graph

.. jupyter-execute::
    :hide-code:

    plot_matched([ex_graphs.node_two_to_one(t) for t in [0, 1]], [{4: "NS", (1, 2): "FN"}, {5: "NS", (1, 7): "FN", (2, 3): "FN"}], "")

Intertrack
^^^^^^^^^^

Intertrack edges connect two nodes with different track IDs, most commonly parent cells to daughter cells, or two cells in non-consecutive frames. Note that intertrack edges are not errors, but support the annotation of Wrong Semantic edges, as detailed below.

.. jupyter-execute::
    :hide-code:

    div_graph = ex_graphs.basic_division(1)
    matched = Matched(div_graph, TrackingGraph(nx.DiGraph()), [], {})
    plot_matched([matched], [{(2, 3): "IT", (2, 4): "IT"}], "")


Wrong Semantic
^^^^^^^^^^^^^^

After identifying a matched pair of edges from the ground truth and predicted graphs, the predicted edge is annotated as wrong semantic if the ground truth and predicted edge have different intertrack edge annotations.

.. jupyter-execute::
    :hide-code:

    plot_matched(
       [ex_graphs.fp_div(1), ex_graphs.one_child(1)],
       [{(6, 7): "WS"}, {(2, 3): "WS"}], 
       ""
    )


Gap-Closing Edges
^^^^^^^^^^^^^^^^^

Gap-closing edges must be identical in the ground truth and predicted graphs in order for no errors to be annotated. Otherwise, the appropriate error is annotated as described above.