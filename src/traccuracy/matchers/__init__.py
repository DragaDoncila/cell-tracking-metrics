"""Subpackage for matching ground truth and prediction tracks

This subpackage contains functions that match nodes and edges
of a ground truth and predicted tracking solution.

Each matching function has the following spec:

Args:
    ground_truth (TrackingData): The ground truth or reference tracking solution
    prediction (TrackingData): A generated tracking solution to be compared
        with the ground truth

Returns:
    list[(gt_node_id, pred_node_id)]: A list of pairs of node_ids, where the first node_id
        refers to a node in ground_truth.tracking_graph, and the second node_id
        refers to a node in prediction.tracking_graph. There is no restriction
        on how many matches a ground truth or prediction node can be a part of,
        although there are helper functions to check if you wish to enforce
        a one-to-one, one-to-many, or many-to-one relationship.

It is assumed that edges are matched if and only if both endpoints are matched.
For example, if (gt_1, pred_1) and (gt_2, pred_2) are in the list of matched nodes,
and edges (gt_1, gt_2) and (pred_1, pred_2) exist, they are also considered matched.

While we specify ground truth and prediction, it is possible to
write a matching function that matches two arbitrary tracking solutions.
"""

from ._base import Matched
from ._compute_overlap import get_labels_with_overlap
from ._ctc import CTCMatcher
from ._iou import IOUMatcher

__all__ = ["CTCMatcher", "IOUMatcher", "Matched", "get_labels_with_overlap"]
