import numpy as np
import pytest

import tests.examples.graphs as ex_graphs
from tests.test_utils import get_division_graphs
from traccuracy import NodeFlag, TrackingGraph
from traccuracy.matchers import Matched
from traccuracy.track_errors.divisions import (
    _classify_divisions,
    _correct_shifted_divisions,
    _evaluate_division_events,
    _get_pred_by_t,
    _get_succ_by_t,
)


class TestStandardsDivisions:
    """Test _classify_divisions against standard cases

    Tests are written for sparse annotations
    """

    @pytest.mark.parametrize("t_div,div_node", [(0, (1, 6)), (1, (2, 6)), (2, (3, 8))])
    def test_good_div(self, t_div, div_node):
        matched = ex_graphs.good_div(t_div)
        _classify_divisions(matched)

        assert matched.gt_graph.nodes[div_node[0]].get(NodeFlag.TP_DIV) is True
        assert matched.pred_graph.nodes[div_node[1]].get(NodeFlag.TP_DIV) is True

    @pytest.mark.parametrize("t_div,pred_div_node", [(0, 6), (1, 6)])
    def test_fp_div(self, t_div, pred_div_node):
        matched = ex_graphs.fp_div(t_div)
        _classify_divisions(matched)

        pred_node_attr = matched.pred_graph.nodes[pred_div_node]
        assert pred_node_attr.get(NodeFlag.FP_DIV) is True

    @pytest.mark.parametrize("t_div,gt_div_node", [(0, 1), (1, 2)])
    def test_one_child(self, t_div, gt_div_node):
        matched = ex_graphs.one_child(t_div)
        _classify_divisions(matched)

        gt_node_attr = matched.gt_graph.nodes[gt_div_node]
        assert gt_node_attr.get(NodeFlag.FN_DIV) is True

    @pytest.mark.parametrize("t_div,gt_div_node", [(0, 1), (1, 2)])
    def test_no_children(self, t_div, gt_div_node):
        matched = ex_graphs.no_children(t_div)
        _classify_divisions(matched)

        gt_node_attr = matched.gt_graph.nodes[gt_div_node]
        assert gt_node_attr.get(NodeFlag.FN_DIV) is True

    @pytest.mark.parametrize("t_div,gt_div_node,pred_div_node", [(0, 1, 8), (1, 2, 7)])
    def test_wrong_child(self, t_div, gt_div_node, pred_div_node):
        matched = ex_graphs.wrong_child(t_div)
        _classify_divisions(matched)

        gt_node_attr = matched.gt_graph.nodes[gt_div_node]
        assert gt_node_attr.get(NodeFlag.FN_DIV) is True

        pred_node_attr = matched.pred_graph.nodes[pred_div_node]
        assert pred_node_attr.get(NodeFlag.FP_DIV) is True


class Test_get_pred_by_t:
    g = ex_graphs.basic_graph()

    def test_predecessor_available(self):
        start_node = 3
        delta = 2
        node = _get_pred_by_t(self.g, start_node, delta)
        assert node == 1

    def test_no_predecessor(self):
        start_node = 2
        delta = 2
        node = _get_pred_by_t(self.g, start_node, delta)
        assert node is None


class Test_get_succ_by_t:
    g = ex_graphs.basic_division(2)

    def across_division(self):
        # Return none if looking across division
        start_node = 3
        delta = 2
        succ = _get_succ_by_t(self.g, start_node, delta)
        assert succ is None

    def valid_succ(self):
        # Find 2 frames forward with valid node
        start_node = 1
        delta = 2
        succ = _get_succ_by_t(self.g, start_node, delta)
        assert succ == 3

    def no_succ(self):
        # Forward without valid node returns None
        start_node = 4
        delta = 1
        succ = _get_succ_by_t(self.g, start_node, delta)
        assert succ is None


class TestStandardShifted:
    """Test correct_shifted_divisions against standard shifted cases"""

    @pytest.mark.parametrize("n_frames", [1, 2])
    @pytest.mark.parametrize(
        "matched, gt_node, pred_node",
        [(ex_graphs.div_1early_end(), 2, 9), (ex_graphs.div_1early_mid(), 3, 9)],
        ids=["div_1early_end", "div_1early_mid"],
    )
    def test_div_1early(self, n_frames, matched, gt_node, pred_node):
        _classify_divisions(matched)
        shifted = _correct_shifted_divisions(matched, n_frames=n_frames)

        attrs = shifted.gt_graph.nodes[gt_node]
        assert attrs.get(NodeFlag.TP_DIV) is True
        assert attrs.get(NodeFlag.FN_DIV) is False

        attrs = shifted.pred_graph.nodes[pred_node]
        assert attrs.get(NodeFlag.TP_DIV) is True
        assert attrs.get(NodeFlag.FP_DIV) is False

    @pytest.mark.parametrize("n_frames", [1, 3])
    @pytest.mark.parametrize(
        "matched, gt_node, pred_node",
        [(ex_graphs.div_2early_end(), 3, 8), (ex_graphs.div_2early_mid(), 4, 8)],
        ids=["div_2early_end", "div_2early_mid"],
    )
    def test_div_2early(self, n_frames, matched, gt_node, pred_node):
        _classify_divisions(matched)
        shifted = _correct_shifted_divisions(matched, n_frames=n_frames)

        if n_frames == 1:  # Not corrected
            attrs = shifted.gt_graph.nodes[gt_node]
            assert attrs.get(NodeFlag.FN_DIV) is True

            attrs = shifted.pred_graph.nodes[pred_node]
            assert attrs.get(NodeFlag.FP_DIV) is True
        elif n_frames == 3:  # corrected
            attrs = shifted.gt_graph.nodes[gt_node]
            assert attrs.get(NodeFlag.TP_DIV) is True
            assert attrs.get(NodeFlag.FN_DIV) is False

            attrs = shifted.pred_graph.nodes[pred_node]
            assert attrs.get(NodeFlag.TP_DIV) is True
            assert attrs.get(NodeFlag.FP_DIV) is False

    @pytest.mark.parametrize("n_frames", [1, 2])
    @pytest.mark.parametrize(
        "matched, gt_node, pred_node",
        [(ex_graphs.div_1late_end(), 1, 11), (ex_graphs.div_1late_mid(), 2, 11)],
        ids=["div_1late_end", "div_1late_mid"],
    )
    def test_div_1late(self, n_frames, matched, gt_node, pred_node):
        _classify_divisions(matched)
        shifted = _correct_shifted_divisions(matched, n_frames=n_frames)

        attrs = shifted.gt_graph.nodes[gt_node]
        assert attrs.get(NodeFlag.TP_DIV) is True
        assert attrs.get(NodeFlag.FN_DIV) is False

        attrs = shifted.pred_graph.nodes[pred_node]
        assert attrs.get(NodeFlag.TP_DIV) is True
        assert attrs.get(NodeFlag.FP_DIV) is False

    @pytest.mark.parametrize("n_frames", [1, 3])
    @pytest.mark.parametrize(
        "matched, gt_node, pred_node",
        [(ex_graphs.div_2late_end(), 1, 12), (ex_graphs.div_2late_mid(), 2, 12)],
        ids=["div_2late_end", "div_2late_mid"],
    )
    def test_div_2late(self, n_frames, matched, gt_node, pred_node):
        _classify_divisions(matched)
        shifted = _correct_shifted_divisions(matched, n_frames=n_frames)

        if n_frames == 1:  # Not corrected
            attrs = shifted.gt_graph.nodes[gt_node]
            assert attrs.get(NodeFlag.FN_DIV) is True

            attrs = shifted.pred_graph.nodes[pred_node]
            assert attrs.get(NodeFlag.FP_DIV) is True
        elif n_frames == 3:  # corrected
            attrs = shifted.gt_graph.nodes[gt_node]
            assert attrs.get(NodeFlag.TP_DIV) is True
            assert attrs.get(NodeFlag.FN_DIV) is False

            attrs = shifted.pred_graph.nodes[pred_node]
            assert attrs.get(NodeFlag.TP_DIV) is True
            assert attrs.get(NodeFlag.FP_DIV) is False

    def test_minimal_matching(self):
        matched = ex_graphs.div_shift_min_match()
        _classify_divisions(matched)
        shifted = _correct_shifted_divisions(matched, n_frames=1)

        attrs = shifted.gt_graph.nodes[2]
        assert attrs.get(NodeFlag.TP_DIV) is True
        assert attrs.get(NodeFlag.FN_DIV) is False

        attrs = shifted.pred_graph.nodes[11]
        assert attrs.get(NodeFlag.TP_DIV) is True
        assert attrs.get(NodeFlag.FP_DIV) is False

    @pytest.mark.parametrize(
        "matched",
        [
            ex_graphs.div_shift_bad_match_pred(),
            ex_graphs.div_shift_bad_match_daughter(),
        ],
        ids=["pred", "daughters"],
    )
    def test_bad_matching(self, matched):
        _classify_divisions(matched)
        shifted = _correct_shifted_divisions(matched, n_frames=1)

        # No correction of shifted divisions b/c matching criteria not met
        attrs = shifted.gt_graph.nodes[2]
        assert attrs.get(NodeFlag.TP_DIV) is None
        assert attrs.get(NodeFlag.FN_DIV) is True

        attrs = shifted.pred_graph.nodes[11]
        assert attrs.get(NodeFlag.TP_DIV) is None
        assert attrs.get(NodeFlag.FP_DIV) is True


def test_evaluate_division_events():
    g_gt, g_pred, map_gt, map_pred = get_division_graphs()
    mapper = list(zip(map_gt, map_pred))
    frame_buffer = 2

    matched_data = Matched(
        TrackingGraph(g_gt), TrackingGraph(g_pred), mapper, {"name": "DummyMatcher"}
    )

    results = _evaluate_division_events(matched_data, max_frame_buffer=frame_buffer)

    assert np.all([isinstance(k, int) for k in results.keys()])
