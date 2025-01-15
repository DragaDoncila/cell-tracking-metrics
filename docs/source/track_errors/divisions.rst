Division Errors
===============
Note: These flags are annotated on the parent nodes.

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
                ax.text(_x - 0.1, _y - 0.25, ann, color='purple', horizontalalignment='right')

        for u, v in graph.graph.edges():
            xs = [graph.graph.nodes[u]["t"], graph.graph.nodes[v]["t"]]
            ys = [graph.graph.nodes[u]["y"], graph.graph.nodes[v]["y"]]
            ax.plot(xs, ys, color=color)

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


    def plot_matched(examples, annotations, suptitle, titles):
        gt_color = "black"
        pred_color = "blue"
        mapping_color = "grey"

        if len(examples) > 1:
            yheight = 2.5
        else:
            yheight = 2

        fig, ax = plt.subplots(1, len(examples) + 1, figsize=(3 * len(examples) + 1, yheight))
        for i, (matched, anns, title) in enumerate(zip(examples, annotations, titles)):
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
            axis.set_title(title)

        handles = [
            Patch(color=gt_color),
            Patch(color=pred_color),
            Patch(color=mapping_color),
            Patch(color='purple')
        ]
        labels = ["Ground Truth", "Prediction", "Mapping", "Node Annotations"]
        ax[-1].legend(handles=handles, labels=labels, loc="center")
        ax[-1].set_axis_off()
        plt.tight_layout()
        fig.suptitle(suptitle, y=1.1)

True Positive
-------------

A true positive division is a division event in which the parent and both daughters match between the ground truth and predicted graphs. True positive divisions are annotated on the parent node on both the ground truth and predicted graphs.

.. jupyter-execute::
    :hide-code:

    plot_matched([ex_graphs.good_div(1)], [{2: "TP", 6: "TP"}], "", [""])

The ``frame_buffer`` parameter allows for divisions to be classified as true positives if they occur within the specified number of frames of tolerance. This feature is useful in cases where the exact frame that a division event occurs is somewhat arbitrary due to a high frame rate or variable segmentation or detection.

For the given ground truth graph, the subsequent predicted graphs show examples of true positive divisions events with different ``frame_buffer`` specifications.

.. jupyter-execute::
    :hide-code:

    plot_matched(
        [ex_graphs.div_1early_mid(), ex_graphs.div_1late_mid()],
        [{9: "TP", 3: "TP"}, {2: "TP", 11: "TP"}],
        "Frame buffer = 1",
        ["Early Division", "Late Division"]
    )

.. jupyter-execute::
    :hide-code:

    plot_matched(
        [ex_graphs.div_2early_mid(), ex_graphs.div_2late_mid()],
        [{8: "TP", 4: "TP"}, {2: "TP", 12: "TP"}],
        "Frame buffer = 2",
        ["Early Division", "Late Division"]
    )

After classifying basic division errors, we consider all false positive and false negative divisions. If a pair of errors occurs within the specified frame buffer, the pair is considered a true positive division if the parent nodes and daughter nodes match. We determine the "parent node" of the late division by traversing back along the graph until we find the node in the same frame as the parent node of the early division. We repeat the process for finding daughters of the early division, by advancing along the graph to find nodes in the same frame as the late division daughters.

False Negative
--------------

A false negative division applies to any case where the parent node is not linked to both correct daughters. False negative divisions are annotated on the ground truth graph.

Given the ground truth graph below, each of the subsequent prediction graphs would be classified as a false negative division.

.. jupyter-execute::
    :hide-code:

    plot_matched(
        [ex_graphs.one_child(1), ex_graphs.no_children(1), ex_graphs.wrong_child(1)],
        [{2: "FN"}, {2: "FN"}, {2: "FN", 7: "FP"}],
        "",
        ["Missing daughter", "Missing daughters", "Wrong daughter"]
    )


False Positive
--------------

A false positive division is any division event in the predicted graph that does not correspond to a division in the ground truth graph. False positive divisions are annotated on the predicted graph.

.. jupyter-execute::
    :hide-code:
    
    plot_matched(
        [ex_graphs.fp_div(1), ex_graphs.wrong_child(1)],
        [{6: "FP"}, {2: "FN", 7: "FP"}],
        "",
        ["No division", "Wrong daughter"]
    )
