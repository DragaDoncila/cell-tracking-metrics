# %% [markdown]
# # Basic Errors
#
# This set of node and edge errors applies only to graphs with a one-to-one matching.

# %% nbsphinx="hidden"
import sys

sys.path.append("../../../tests")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import examples.graphs as ex_graphs
from traccuracy._tracking_graph import TrackingGraph
from traccuracy.matchers import Matched


def get_loc(graph, node):
    return graph.graph.nodes[node]["t"], graph.graph.nodes[node]["y"]


def plot_graph(
    ax, graph: TrackingGraph, color="black", annotations={}, ann_color="red"
):
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
            ax.text(
                _x - 0.1, _y - 0.25, ann, color="purple", horizontalalignment="right"
            )

    for u, v in graph.graph.edges():
        xs = [graph.graph.nodes[u]["t"], graph.graph.nodes[v]["t"]]
        ys = [graph.graph.nodes[u]["y"], graph.graph.nodes[v]["y"]]
        ax.plot(xs, ys, color=color)
        # Plot edge annotation if available
        ann = annotations.get((u, v))
        if ann:
            xx = sum(xs) / 2
            yy = sum(ys) / 2
            ax.text(xx + 0.1, yy, ann, color="orange", horizontalalignment="center")

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

    fig, ax = plt.subplots(
        1, len(examples) + 1, figsize=(3 * len(examples) + 1, yheight)
    )
    for i, (matched, anns, title) in enumerate(zip(examples, annotations, titles)):
        axis = ax[i]
        xbounds, ybounds = plot_graph(
            axis, matched.gt_graph, color=gt_color, annotations=anns
        )
        bounds = plot_graph(
            axis, matched.pred_graph, color=pred_color, annotations=anns
        )
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
        Patch(color="purple"),
        Patch(color="orange"),
    ]
    labels = [
        "Ground Truth",
        "Prediction",
        "Mapping",
        "Node Annotations",
        "Edge Annotation",
    ]
    ax[-1].legend(handles=handles, labels=labels, loc="center")
    ax[-1].set_axis_off()
    plt.tight_layout()
    fig.suptitle(suptitle, y=1.1)
    plt.show()


# %% [markdown]
# ## Nodes
# ### True Positive
# A true positive node is defined as a predicted node that matches to only one
# ground truth node. Additionally, the corresponding ground truth node cannot be
# matched to more than one predicted node. True positives are annotated on both
# the ground truth and the division graph.

# %% [markdown]
# ### False Positive
# A false positive node is a node on the predicted graph does not match to a
# node on the ground truth graph. False positives are annotated on the predicted
# graph.

# %%
plot_matched([ex_graphs.fp_edge_matched(1)], [{7: "FP", 8: "FP"}], "", [""])

# %% [markdown]
# ### False Negative
# A false negative node is a node on the ground truth graph that is not matched
# to a predicted node. False negatives are annotated on the ground truth graph.

# %%
plot_matched([ex_graphs.fn_node_matched(2)], [{3: "FN"}], "", [""])


# %% [markdown]
# ## Edges
# ### True Positive
# An edge in the ground truth is a true positive edge if both source and target
# node are true positives and the corresponding edge is present in the
# prediction. True positive edges are annotated on both the ground truth and the
# predicted graph.

# %% [markdown]
# ### False Positive
# False positive edges occur in and are annotated on the predicted graph. A
# false positive edge can occur in several different scenarios when the edge
# fails to meet the criteria for a true positive.
#
# - Ex 1: Nodes 7 and 8 are not matched to any node in the GT.
# - Ex 2: While Nodes 5 and 6 are matched to nodes in the GT, an edge does not
#   exist between the corresponding GT nodes 2 and 3.
# - Ex 3: Node 6 is not matched to a node in the GT.


# %% nbsphinx="hidden"
def fp_node_match():
    matched = ex_graphs.good_matched()
    matched.gt_graph.graph.remove_edge(2, 3)
    return matched


def fp_node_not_matched():
    matched = ex_graphs.good_matched()
    mapping = matched.mapping
    mapping.remove((3, 6))
    return Matched(matched.gt_graph, matched.pred_graph, mapping, {})


# %%
plot_matched(
    [
        ex_graphs.fp_edge_matched(1),
        fp_node_match(),
        fp_node_not_matched(),
    ],
    [{(7, 8): "FP"}, {(5, 6): "FP"}, {(5, 6): "FP"}],
    "",
    ["Ex. 1", "Ex. 2", "Ex. 3"],
)

# %% [markdown]
# ### False Negative
# A false negative edge can occur in the ground truth graph when:
#
# - One or both of the nodes is not a true positive and matched to a node in the
#   prediction
# - The matched nodes in the prediction do not also have an edge between them
# %%
plot_matched(
    [
        ex_graphs.fn_edge_matched(0),
        ex_graphs.crossover_edge(),
        ex_graphs.fn_node_matched(0),
    ],
    [{(1, 2): "FN"}, {(1, 2): "FN", (4, 5): "FN", (5, 6): "FN"}, {(1, 2): "FN"}],
    "",
    ["", "", ""],
)

# %% [markdown]
# ### Gap-Closing Edges
# Gap-closing edges will be annotated as False Positive on the predicted graph
# and False Negative on the ground truth graph unless both graphs contain an
# identical gap-closing edge. Similarly, gap-closing edges on the predicted
# graph that do not exist in the ground truth graph will be annotated as False
# Positive, with corresponding False Negative annotations on the ground truth.
#
# - Ex 1: The gap-closing edges are identical, they will be marked TP.
# - Ex 2: The GT gap-closing edge is not present in the prediction. It is
#   annotated as FN. Predicted edges are FP.
# - Ex 3: The predicted gap-closing edge is not present in the GT. It is
#  annotated as FP. GT edges are FN.
# %%
plot_matched(
    [
        ex_graphs.gap_close_matched_gap(),
        ex_graphs.gap_close_gt_gap(),
        ex_graphs.gap_close_pred_gap(),
    ],
    [
        {},
        {(1, 3): "FN", (5, 6): "FP", (6, 7): "FP"},
        {(2, 3): "FN", (3, 4): "FN", (6, 8): "FP"},
    ],
    "",
    ["Ex. 1", "Ex. 2", "Ex. 3"],
)
