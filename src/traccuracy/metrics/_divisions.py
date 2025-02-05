"""This submodule classifies division errors in tracking graphs

Each division is classified as one of the following:
- true positive
- false positive
- false negative

These functions require two `TrackingGraph` objects and a mapper between
nodes in the two graphs. Divisions are identified as correct if both the parent
and daughter nodes match between the GT and predicted graph.

Temporal tolerance for correct divisions is implemented to allow for cases in
which the exact frame that a division event occurs is somewhat arbitrary due to
a high frame rate or variable segmentation or detection. Consider the following
graphs as an example::

    G1
                                2_4
    1_0 -- 1_1 -- 1_2 -- 1_3 -<
                                3_4
    G2
                  2_2 -- 2_3 -- 2_4
    1_0 -- 1_1 -<
                  3_2 -- 3_3 -- 3_4

After classifying basic division errors, we consider all false positive and false
negative divisions. If a pair of errors occurs within the specified frame buffer,
the pair is considered a true positive division if the parent nodes and daughter
nodes match. We determine the "parent node" of the late division, e.g. node 1_3 in
graph G1, by traversing back along the graph until we find the node in the same frame
as the parent node of the early division. We repeat the process for finding daughters
of the early division, by advancing along the graph to find nodes in the same frame
as the late division daughters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from traccuracy._tracking_graph import NodeFlag
from traccuracy.matchers._base import Matched
from traccuracy.track_errors.divisions import _evaluate_division_events

from ._base import Metric

if TYPE_CHECKING:
    from traccuracy.matchers import Matched


class DivisionMetrics(Metric):
    """Classify division events and provide the following summary metrics

    - Division Recall
    - Division Precision
    - Division F1 Score
    - Mitotic Branching Correctness: TP / (TP + FP + FN) as defined by Ulicna, K.,
        Vallardi, G., Charras, G. & Lowe, A. R. Automated deep lineage tree analysis
        using a Bayesian single cell tracking approach. Frontiers in Computer Science
        3, 734559 (2021).

    Args:
        max_frame_buffer (int, optional): Maximum value of frame buffer to use in correcting
            shifted divisions. Divisions will be evaluated for all integer values of frame
            buffer between 0 and max_frame_buffer
    """

    def __init__(self, max_frame_buffer=0):
        valid_matching_types = ["one-to-one"]
        super().__init__(valid_matching_types)

        self.frame_buffer = max_frame_buffer

    def _compute(self, data: Matched):
        """Runs `_evaluate_division_events` and calculates summary metrics for each frame buffer

        Args:
            matched_data (traccuracy.matchers.Matched): Matched object for set of GT and Pred data
                Must meet the `needs_one_to_one` criteria

        Returns:
            dict: Returns a nested dictionary with one dictionary per frame buffer value
        """
        div_annotations = _evaluate_division_events(
            data,
            max_frame_buffer=self.frame_buffer,
        )

        return {
            f"Frame Buffer {fb}": self._calculate_metrics(
                matched_data.gt_graph,
                matched_data.pred_graph,
            )
            for fb, matched_data in div_annotations.items()
        }

    def _calculate_metrics(self, g_gt, g_pred):
        tp_division_count = len(g_gt.get_nodes_with_flag(NodeFlag.TP_DIV))
        fn_division_count = len(g_gt.get_nodes_with_flag(NodeFlag.FN_DIV))
        fp_division_count = len(g_pred.get_nodes_with_flag(NodeFlag.FP_DIV))
        wc_division_count = len(g_pred.get_nodes_with_flag(NodeFlag.WC_DIV))

        try:
            recall = tp_division_count / (
                tp_division_count + fn_division_count + wc_division_count
            )
        except ZeroDivisionError:
            recall = 0

        try:
            precision = tp_division_count / (
                tp_division_count + fp_division_count + wc_division_count
            )
        except ZeroDivisionError:
            precision = 0

        try:
            f1 = 2 * (recall * precision) / (recall + precision)
        except ZeroDivisionError:
            f1 = 0

        try:
            mbc = tp_division_count / (
                tp_division_count
                + fn_division_count
                + fp_division_count
                + wc_division_count
            )
        except ZeroDivisionError:
            mbc = 0

        return {
            "Division Recall": recall,
            "Division Precision": precision,
            "Division F1": f1,
            "Mitotic Branching Correctness": mbc,
            "True Positive Divisions": tp_division_count,
            "False Positive Divisions": fp_division_count,
            "False Negative Divisions": fn_division_count,
            "Wrong Children Divisions": wc_division_count,
        }
