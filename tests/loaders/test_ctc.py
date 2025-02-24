import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from numpy.testing import assert_array_equal

from traccuracy._tracking_graph import TrackingGraph
from traccuracy.loaders import _ctc, _load_tiffs


def test_ctc_to_graph():
    # cell_id, start, end, parent_id
    data = [
        {"Cell_ID": 1, "Start": 0, "End": 3, "Parent_ID": 0},
        {"Cell_ID": 2, "Start": 0, "End": 2, "Parent_ID": 0},
        {"Cell_ID": 3, "Start": 3, "End": 3, "Parent_ID": 2},
        {"Cell_ID": 4, "Start": 3, "End": 3, "Parent_ID": 2},
        {"Cell_ID": 5, "Start": 3, "End": 3, "Parent_ID": 4},
    ]
    df = pd.DataFrame(data)
    G = _ctc.ctc_to_graph(
        df, pd.DataFrame({"segmentation_id": [], "x": [], "y": [], "z": [], "t": []})
    )
    for d in data:
        node_ids = [
            "{}_{}".format(d["Cell_ID"], t) for t in range(d["Start"], d["End"] + 1)
        ]

        for node_id in node_ids:
            assert node_id in G

        if d["Parent_ID"]:  # should have a division
            parent_row = df[df["Cell_ID"] == d["Parent_ID"]].iloc[
                0
            ]  # Needs to be a Series for indexing below
            daughter_id = "{}_{}".format(d["Cell_ID"], d["Start"])
            parent_id = "{}_{}".format(parent_row["Cell_ID"], parent_row["End"])
            if G.has_node(parent_id):
                assert G.has_edge(parent_id, daughter_id)
            else:
                assert not G.in_degree(daughter_id)


def test_ctc_single_nodes():
    data = [
        {"Cell_ID": 1, "Start": 0, "End": 3, "Parent_ID": 0},
        {"Cell_ID": 2, "Start": 0, "End": 2, "Parent_ID": 0},
        {"Cell_ID": 3, "Start": 3, "End": 3, "Parent_ID": 2},
        {"Cell_ID": 4, "Start": 3, "End": 3, "Parent_ID": 2},
        {"Cell_ID": 5, "Start": 3, "End": 3, "Parent_ID": 4},
    ]

    detections = [
        {"segmentation_id": 1, "x": 1, "y": 2, "z": 1, "t": 0},
        {"segmentation_id": 1, "x": 1, "y": 2, "z": 1, "t": 1},
        {"segmentation_id": 1, "x": 1, "y": 2, "z": 1, "t": 2},
        {"segmentation_id": 1, "x": 1, "y": 2, "z": 1, "t": 3},
        {"segmentation_id": 2, "x": 2, "y": 1, "z": 1, "t": 0},
        {"segmentation_id": 2, "x": 2, "y": 1, "z": 1, "t": 1},
        {"segmentation_id": 2, "x": 2, "y": 1, "z": 1, "t": 2},
        {"segmentation_id": 2, "x": 2, "y": 1, "z": 1, "t": 3},
        {"segmentation_id": 3, "x": 1, "y": 1, "z": 1, "t": 3},
        {"segmentation_id": 4, "x": 1, "y": 2, "z": 2, "t": 3},
        {"segmentation_id": 5, "x": 1, "y": 1, "z": 2, "t": 3},
    ]
    df = pd.DataFrame(data)
    G = _ctc.ctc_to_graph(df, pd.DataFrame.from_records(detections))
    # This should raise an error if there are no times for single nodes
    TrackingGraph(G)


def test_ctc_with_gap_closing():
    data = [
        {"Cell_ID": 1, "Start": 0, "End": 1, "Parent_ID": 0},
        {"Cell_ID": 2, "Start": 0, "End": 1, "Parent_ID": 0},
        # Connecting frame 1 to frame 3
        {"Cell_ID": 3, "Start": 3, "End": 5, "Parent_ID": 1},
        # Connecting frame 1 to frame 6
        {"Cell_ID": 4, "Start": 6, "End": 8, "Parent_ID": 2},
    ]
    df = pd.DataFrame(data)
    G = _ctc.ctc_to_graph(
        df, pd.DataFrame({"segmentation_id": [], "x": [], "y": [], "z": [], "t": []})
    )
    assert G.has_edge("1_1", "3_3")
    assert G.has_edge("2_1", "4_6")


def test_load_data():
    test_dir = os.path.abspath(__file__)
    data_dir = os.path.abspath(
        os.path.join(test_dir, "../../../examples/sample-data/Fluo-N2DL-HeLa/01_RES/")
    )
    track_path = os.path.join(data_dir, "res_track.txt")
    track_data = _ctc.load_ctc_data(data_dir, track_path)
    assert isinstance(track_data, TrackingGraph)
    assert len(track_data.segmentation) == 92


def test_load_data_no_track_path():
    test_dir = os.path.abspath(__file__)
    data_dir = os.path.abspath(
        os.path.join(test_dir, "../../../examples/sample-data/Fluo-N2DL-HeLa/01_RES/")
    )
    track_data = _ctc.load_ctc_data(data_dir)
    assert isinstance(track_data, TrackingGraph)
    assert len(track_data.segmentation) == 92


def test_load_tiffs_float_data(tmp_path):
    test_dir = os.path.abspath(__file__)
    data_dir = os.path.abspath(
        os.path.join(test_dir, "../../../examples/sample-data/Fluo-N2DL-HeLa/01_RES/")
    )

    files = glob.glob(f"{data_dir}/*.tif*")
    for file in files:
        arr = tifffile.imread(file).astype(np.float64)
        tifffile.imwrite(tmp_path / Path(file).name, arr)
    with pytest.warns(UserWarning, match="Segmentation has float64: casting to uint64"):
        casted_seg = _load_tiffs(tmp_path)
    orig_seg = _load_tiffs(data_dir)
    assert casted_seg.dtype == np.uint64
    assert_array_equal(casted_seg.astype(orig_seg.dtype), orig_seg)
