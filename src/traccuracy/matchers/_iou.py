from __future__ import annotations

from typing import Hashable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from traccuracy._tracking_graph import TrackingGraph

from ._base import Matcher
from ._compute_overlap import get_labels_with_overlap


def _match_nodes(
    gt: np.ndarray, res: np.ndarray, threshold: float = 0.5, one_to_one: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Identify overlapping objects according to IoU and a threshold for minimum overlap.

    QUESTION: Does this rely on sequential segmentation labels

    Args:
        gt (np.ndarray): labeled frame
        res (np.ndarray): labeled frame
        threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
            If segmentations are identical, 1 works well.
            For imperfect segmentations try 0.6-0.8 to get better matching
        one_to_one (optional, bool): If True, forces the mapping to be one-to-one by running
            linear assignment on the thresholded iou array. Default False.

    Returns:
        gtcells (np.ndarray): Array of overlapping ids in the gt frame.
        rescells (np.ndarray): Array of overlapping ids in the res frame.
    """
    if threshold == 0.0 and not one_to_one:
        raise ValueError("Threshold of 0 is not valid unless one_to_one is True")
    # casting to int to avoid issue #152 (result is float with numpy<2, dtype=uint64)
    iou = np.zeros((int(np.max(gt) + 1), int(np.max(res) + 1)))

    ious = get_labels_with_overlap(gt, res, overlap="iou")

    for gt_label, res_label, iou_val in ious:
        if iou_val >= threshold:
            iou[gt_label, res_label] = iou_val

    if one_to_one:
        pairs = _one_to_one_assignment(iou)
    else:
        pairs = np.where(iou)

    # Catch the case where there are no overlaps
    if len(pairs) < 2:
        gtcells, rescells = [], []
    else:
        gtcells, rescells = pairs[0], pairs[1]

    return gtcells, rescells


def _one_to_one_assignment(iou, unmapped_cost=4):
    """Perform linear assignment on the iou matrix to create a one-to-one
    mapping

    Args:
        iou (np.array): Array containing thresholded iou values
        unmapped_cost (float, optional): Cost of an unassigned cell.
            Lower values leads to more unassigned cells. Defaults to 4.

    Returns:
        tuple: Tuple of two arrays, one for indices of each axis
    """
    # Determine number of objects in zeroth and first axis
    # Exclude the background which is currently included in iou matrix
    n0 = iou.shape[0] - 1
    n1 = iou.shape[1] - 1
    n_obj = n0 + n1
    matrix = np.ones((n_obj, n_obj))

    # Assign 1 - iou to top left and bottom right
    cost = 1 - iou[1:, 1:]
    # increase the cost for those with no IOU to higher than the unmapped cost
    cost[cost == 1] = unmapped_cost + 1
    matrix[:n0, :n1] = cost
    matrix[n_obj - n1 :, n_obj - n0 :] = cost.T

    # Calculate unassigned corners, with base cost of 10*unmapped cost
    # Diagonals are set to unmapped_cost
    bl = np.full((n1, n1), unmapped_cost * 10)
    np.fill_diagonal(bl, unmapped_cost)
    tr = np.full((n0, n0), unmapped_cost * 10)
    np.fill_diagonal(tr, unmapped_cost)

    # Assign diagonals to cm
    matrix[n_obj - n1 :, :n1] = bl
    matrix[:n0, n_obj - n0 :] = tr

    results = linear_sum_assignment(matrix)

    # Map results back to cost matrix
    assignment_matrix = np.zeros_like(matrix)
    assignment_matrix[results] = 1

    # Pull out only the direct matches from the top left corner
    matches = np.nonzero(assignment_matrix[:n0, :n1])

    # Add 1 to all indices to correct for removing the background previously
    matches = (matches[0] + 1, matches[1] + 1)
    return matches


def _construct_time_to_seg_id_map(
    graph: TrackingGraph,
) -> dict[int, dict[Hashable, Hashable]]:
    """For each time frame in the graph, create a mapping from segmentation ids
    (the ids in the segmentation array, stored in graph.label_key) to the
    node ids (the ids of the TrackingGraph nodes).

    Args:
        graph(TrackingGraph): a tracking graph with a label_key on each node

    Returns:
      dict[int, dict[Hashable, Hashable]]: a dictionary from {time: {segmentation_id: node_id}}

    Raises:
        AssertionError: If two nodes in a time frame have the same segmentation_id
    """
    time_to_seg_id_map: dict[int, dict[Hashable, Hashable]] = {}
    for node_id, data in graph.nodes(data=True):
        time = data[graph.frame_key]
        seg_id = data[graph.label_key]
        seg_id_to_node_id_map = time_to_seg_id_map.get(time, {})
        assert (
            seg_id not in seg_id_to_node_id_map
        ), f"Segmentation ID {seg_id} occurred twice in frame {time}."
        seg_id_to_node_id_map[seg_id] = node_id
        time_to_seg_id_map[time] = seg_id_to_node_id_map
    return time_to_seg_id_map


def match_iou(gt, pred, threshold=0.6, one_to_one=False):
    """Identifies pairs of cells between gt and pred that have iou > threshold

    This can return more than one match for any node
    Assumes that within a frame, each object has a unique segmentation label
    and that the label is recorded on each node using label_key

    Args:
        gt (traccuracy.TrackingGraph): Tracking data object containing graph and segmentations
        pred (traccuracy.TrackingGraph): Tracking data object containing graph and segmentations
        threshold (float, optional): Minimum IoU for matching cells. Defaults to 0.6.
        one_to_one (optional, bool): If True, forces the mapping to be one-to-one by running
            linear assignment on the thresholded iou array. Default False.

    Returns:
        list[(gt_node, pred_node)]: list of tuples where each tuple contains a gt node and pred node

    Raises:
        ValueError: gt and pred must be a TrackingData object
        ValueError: GT and pred segmentations must be the same shape
    """
    if not isinstance(gt, TrackingGraph) or not isinstance(pred, TrackingGraph):
        raise ValueError(
            "Input data must be a TrackingData object with a graph and segmentations"
        )

    mapper = []

    if gt.segmentation.shape != pred.segmentation.shape:
        raise ValueError("Segmentation shapes must match between gt and pred")

    # Get overlaps for each frame
    frame_range = range(gt.start_frame, gt.end_frame)
    total = len(list(frame_range))

    gt_time_to_seg_id_map = _construct_time_to_seg_id_map(gt)
    pred_time_to_seg_id_map = _construct_time_to_seg_id_map(pred)

    for i, t in tqdm(enumerate(frame_range), desc="Matching frames", total=total):
        matches = _match_nodes(
            gt.segmentation[i],
            pred.segmentation[i],
            threshold=threshold,
            one_to_one=one_to_one,
        )
        # Construct node id tuple for each match
        for gt_seg_id, pred_seg_id in zip(*matches):
            # Find node id based on time and segmentation label
            gt_node = gt_time_to_seg_id_map[t][gt_seg_id]
            pred_node = pred_time_to_seg_id_map[t][pred_seg_id]
            mapper.append((gt_node, pred_node))
    return mapper


class IOUMatcher(Matcher):
    """Constructs a mapping between gt and pred nodes using the IoU of the segmentations

    Lower values for iou_threshold will be more permissive of imperfect matches

    Args:
        iou_threshold (float, optional): Minimum IoU value to assign a match. Defaults to 0.6.
        one_to_one (optional, bool): If True, forces the mapping to be one-to-one by running
            linear assignment on the thresholded iou array. Default False.
    """

    def __init__(self, iou_threshold=0.6, one_to_one=False):
        self.iou_threshold = iou_threshold
        self.one_to_one = one_to_one

        # If either condition is met, matching must be one to one
        if one_to_one or iou_threshold > 0.5:
            self._matching_type = "one-to-one"

    def _compute_mapping(self, gt_graph: TrackingGraph, pred_graph: TrackingGraph):
        """Computes IOU mapping for a set of grpahs

        Args:
            gt_graph (TrackingGraph): Tracking graph object for the gt with segmentation data
            pred_graph (TrackingGraph): Tracking graph object for the pred with segmentation data

        Raises:
            ValueError: Segmentation data must be provided for both gt and pred data

        Returns:
            Matched: Matched data object containing IOU mapping
        """
        # Check that segmentations exist in the data
        if gt_graph.segmentation is None or pred_graph.segmentation is None:
            raise ValueError(
                "Segmentation data must be provided for both gt and pred data"
            )

        mapping = match_iou(
            gt_graph,
            pred_graph,
            threshold=self.iou_threshold,
            one_to_one=self.one_to_one,
        )

        return mapping
