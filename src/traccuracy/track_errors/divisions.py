from __future__ import annotations

import copy
import itertools
import logging
from collections import Counter
from typing import TYPE_CHECKING

from traccuracy._tracking_graph import NodeFlag
from traccuracy._utils import find_gt_node_matches, find_pred_node_matches

if TYPE_CHECKING:
    from traccuracy.matchers import Matched

logger = logging.getLogger(__name__)


def _classify_divisions(matched_data: Matched):
    """Identify each division as a true positive, false positive or false negative

    This function only works on node mappers that are one-to-one

    Graphs are annotated in place and therefore not returned

    Args:
        matched_data (Matched): Matched data object containing gt and pred graphs
            with their associated mapping

    Raises:
        ValueError: mapper must contain a one-to-one mapping of nodes
    """
    g_gt = matched_data.gt_graph
    g_pred = matched_data.pred_graph
    mapper = matched_data.mapping

    if g_gt.division_annotations and g_pred.division_annotations:
        logger.info("Division annotations already present. Skipping graph annotation.")
        return

    # Check that mapper is one to one
    if len(mapper) != len({pair[0] for pair in mapper}) or len(mapper) != len(
        {pair[1] for pair in mapper}
    ):
        raise ValueError("Mapping must be one-to-one")

    def _find_gt_node_matches(gt_node):
        match = find_gt_node_matches(mapper, gt_node)
        if len(match) > 0:
            return match[0]

    def _find_pred_node_matches(pred_node):
        match = find_pred_node_matches(mapper, pred_node)
        if len(match) > 0:
            return match[0]

    # Collect list of divisions
    div_gt = g_gt.get_divisions()
    div_pred = g_pred.get_divisions()

    for gt_node in div_gt:
        # Find possible matching nodes
        pred_node = _find_gt_node_matches(gt_node)
        # No matching node so division missed
        if pred_node is None:
            g_gt.set_flag_on_node(gt_node, NodeFlag.FN_DIV, True)
        # Check if the division has the correct daughters
        else:
            succ_gt = g_gt.graph.successors(gt_node)
            # Map pred succ nodes onto gt, unmapped nodes will return as None
            succ_pred = [
                _find_pred_node_matches(n) for n in g_pred.graph.successors(pred_node)
            ]

            # If daughters are same, division is correct
            cnt_gt = Counter(succ_gt)
            cnt_pred = Counter(succ_pred)
            if cnt_gt == cnt_pred:
                g_gt.set_flag_on_node(gt_node, NodeFlag.TP_DIV, True)
                g_pred.set_flag_on_node(pred_node, NodeFlag.TP_DIV, True)
            # Only one daughter matched
            elif len(cnt_gt - cnt_pred) == 1:
                g_gt.set_flag_on_node(gt_node, NodeFlag.FN_DIV)
                g_pred.set_flag_on_node(pred_node, NodeFlag.FP_DIV)
            # If daughters are at all mismatched, division is false negative
            else:
                g_gt.set_flag_on_node(gt_node, NodeFlag.FN_DIV, True)

        # Remove res division to record that we have classified it
        if pred_node in div_pred:
            div_pred.remove(pred_node)

    # Any remaining pred divisions are false positives
    for fp_div in div_pred:
        g_pred.set_flag_on_node(fp_div, NodeFlag.FP_DIV, True)

    # Set division annotation flag
    g_gt.division_annotations = True
    g_pred.division_annotations = True


def _get_pred_by_t(g, node, delta_frames):
    """For a given graph and node, traverses back by predecessor until target_frame

    Args:
        G (TrackingGraph): TrackingGraph to search on
        node (hashable): Key of starting node
        target_frame (int): Frame of the predecessor target node

    Raises:
        ValueError: Cannot operate on graphs with merges

    Returns:
        hashable: Node key of predecessor in target frame
    """
    for _ in range(delta_frames):
        nodes = list(g.graph.predecessors(node))
        # Exit if there are no predecessors
        if len(nodes) == 0:
            return None
        # Fail if finding merges
        elif len(nodes) > 1:
            raise ValueError("Cannot operate on graphs with merges")
        node = nodes[0]

    return node


def _get_succ_by_t(g, node, delta_frames):
    """For a given node, find the successors after delta frames

    If a division event is discovered, returns None

    Args:
        G (TrackingGraph): TrackingGraph to search on
        node (hashable): Key of starting node
        target_frame (int): Frame of the successor target node

    Returns:
        hashable: Node id of successor
    """
    for _ in range(delta_frames):
        nodes = list(g.graph.successors(node))
        # Exit if there are no successors another division
        if len(nodes) == 0 or len(nodes) >= 2:
            return None
        node = nodes[0]

    return node


def _correct_shifted_divisions(matched_data: Matched, n_frames=1):
    """Allows for divisions to occur within a frame buffer and still be correct

    This implementation asserts that the parent lineages and daughter lineages must match.
    Matching is determined based on the provided mapper
    Does not support merges

    Copies matched_data before modifying node annotations and returns the new versions

    Args:
        matched_data (Matched): Matched data object containing gt and pred graphs
            with their associated mapping
        n_frames (int): Number of frames to include in the frame buffer

    Returns:
        Matched: copy of matched_data with corrected division annotations
    """
    # Create copies of the graphs to modify during correction of divisions
    new_matched = copy.deepcopy(matched_data)
    g_gt = new_matched.gt_graph
    g_pred = new_matched.pred_graph
    mapper = new_matched.mapping

    # Check that mapper is one to one
    if len(mapper) != len({pair[0] for pair in mapper}) or len(mapper) != len(
        {pair[1] for pair in mapper}
    ):
        raise ValueError("Mapping must be one-to-one")

    fp_divs = g_pred.get_nodes_with_flag(NodeFlag.FP_DIV)
    fn_divs = g_gt.get_nodes_with_flag(NodeFlag.FN_DIV)

    gt_to_pred_dict = dict(mapper)
    pred_to_gt_dict = {pred: gt for (gt, pred) in mapper}

    # Compare all pairs of fp and fn
    for fp_node, fn_node in itertools.product(fp_divs, fn_divs):
        correct = False
        t_fp = g_pred.graph.nodes[fp_node][g_pred.frame_key]
        t_fn = g_gt.graph.nodes[fn_node][g_gt.frame_key]

        # Move on if nodes are not within frame buffer or within same frame
        if abs(t_fp - t_fn) > n_frames or t_fp == t_fn:
            continue

        # False positive in pred occurs before false negative in gt
        if t_fp < t_fn:
            # Check if fp node matches predecessor of fn
            fn_pred = _get_pred_by_t(g_gt, fn_node, t_fn - t_fp)
            # Check if the match exists
            if (fn_pred, fp_node) not in mapper:
                # Match does not exist so divisions cannot match
                continue

            # Check if daughters match
            fp_succ = [
                _get_succ_by_t(g_pred, node, t_fn - t_fp)
                for node in g_pred.graph.successors(fp_node)
            ]
            fn_succ = g_gt.graph.successors(fn_node)
            fn_succ_mapped = [
                gt_to_pred_dict[fn] for fn in fn_succ if fn in gt_to_pred_dict
            ]
            if Counter(fp_succ) != Counter(fn_succ_mapped):
                # Daughters don't match so division cannot match
                continue

            # At this point daughters and parents match so division is correct
            correct = True
        # False negative in gt occurs before false positive in pred
        else:
            # Check if fp node matches fn predecessor
            fp_pred = _get_pred_by_t(g_pred, fp_node, t_fp - t_fn)
            # Check if match exists
            if (fn_node, fp_pred) not in mapper:
                # Match does not exist so divisions cannot match
                continue

            # Check if daughters match
            fn_succ = [
                _get_succ_by_t(g_gt, node, t_fp - t_fn)
                for node in g_gt.graph.successors(fn_node)
            ]
            fp_succ = g_pred.graph.successors(fp_node)

            fp_succ_mapped = [
                pred_to_gt_dict[fp] for fp in fp_succ if fp in pred_to_gt_dict
            ]
            if Counter(fp_succ_mapped) != Counter(fn_succ):
                # Daughters don't match so division cannot match
                continue

            # At this point daughters and parents match so division is correct
            correct = True

        if correct:
            # Remove error annotations from pred graph
            g_pred.set_flag_on_node(fp_node, NodeFlag.FP_DIV, False)
            g_gt.set_flag_on_node(fn_node, NodeFlag.FN_DIV, False)

            # Add the tp divisions annotations
            g_gt.set_flag_on_node(fn_node, NodeFlag.TP_DIV, True)
            g_pred.set_flag_on_node(fp_node, NodeFlag.TP_DIV, True)

    return new_matched


def _evaluate_division_events(matched_data: Matched, max_frame_buffer=0):
    """Classify division errors and correct shifted divisions according to frame_buffer

    Note: A copy of matched_data will be created for each frame_buffer other than 0.
    For large graphs, creating copies may introduce memory problems.

    Args:
        matched_data (Matched): Matched data object containing gt and pred graphs
            with their associated mapping
        max_frame_buffer (int, optional): Maximum value of frame buffer to use in correcting
            shifted divisions. Divisions will be evaluated for all integer values of frame
            buffer between 0 and max_frame_buffer

    Returns:
        dict {frame_buffer: matched_data}: A dictionary where each key corresponds to a frame
            buffer with a tuple of the corresponding ground truth and predicted TrackingGraphs
            after division annotations and correction by frame buffer
    """
    div_annotations = {}

    # Baseline division classification
    _classify_divisions(matched_data)
    div_annotations[0] = matched_data

    # Correct shifted divisions for each nonzero value in frame_buffer
    for delta in range(1, max_frame_buffer + 1):
        corrected_matched = _correct_shifted_divisions(matched_data, n_frames=delta)
        div_annotations[delta] = corrected_matched

    return div_annotations
