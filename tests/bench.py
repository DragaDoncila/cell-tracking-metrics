import copy
import os
import urllib.request
import zipfile

import pandas as pd
import pytest
from traccuracy.loaders import (
    _check_ctc,
    _get_node_attributes,
    _load_tiffs,
    load_ctc_data,
)
from traccuracy.matchers import CTCMatcher, IOUMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_gt_data():
    # Download GT data -- look into caching this in github actions
    url = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip"
    data_dir = os.path.join(ROOT_DIR, "downloads")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    filename = url.split("/")[-1]
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

        # Unzip the data
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)


@pytest.fixture(scope="module")
def gt_data():
    download_gt_data()
    return load_ctc_data(
        os.path.join(ROOT_DIR, "downloads/Fluo-N2DL-HeLa/01_GT/TRA"),
        os.path.join(ROOT_DIR, "downloads/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"),
        run_checks=False,
    )


@pytest.fixture(scope="module")
def pred_data():
    return load_ctc_data(
        os.path.join(ROOT_DIR, "examples/sample-data/Fluo-N2DL-HeLa/01_RES"),
        os.path.join(
            ROOT_DIR, "examples/sample-data/Fluo-N2DL-HeLa/01_RES/res_track.txt"
        ),
        run_checks=False,
    )


@pytest.fixture(scope="module")
def ctc_matched(gt_data, pred_data):
    return CTCMatcher().compute_mapping(gt_data, pred_data)


@pytest.fixture(scope="module")
def iou_matched(gt_data, pred_data):
    return IOUMatcher(iou_threshold=0.1).compute_mapping(gt_data, pred_data)


def test_load_gt_data(benchmark):
    download_gt_data()

    benchmark.pedantic(
        load_ctc_data,
        args=(
            "downloads/Fluo-N2DL-HeLa/01_GT/TRA",
            "downloads/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt",
        ),
        kwargs={"run_checks": False},
        rounds=1,
        iterations=1,
    )


def test_load_pred_data(benchmark):
    benchmark.pedantic(
        load_ctc_data,
        args=(
            os.path.join(ROOT_DIR, "examples/sample-data/Fluo-N2DL-HeLa/01_RES"),
            os.path.join(
                ROOT_DIR, "examples/sample-data/Fluo-N2DL-HeLa/01_RES/res_track.txt"
            ),
        ),
        kwargs={"run_checks": False},
        rounds=1,
        iterations=1,
    )


def test_ctc_checks(benchmark):
    names = ["Cell_ID", "Start", "End", "Parent_ID"]

    tracks = pd.read_csv(
        os.path.join(
            ROOT_DIR, "examples/sample-data/Fluo-N2DL-HeLa/01_RES/res_track.txt"
        ),
        header=None,
        sep=" ",
        names=names,
    )

    masks = _load_tiffs(
        os.path.join(ROOT_DIR, "examples/sample-data/Fluo-N2DL-HeLa/01_RES")
    )
    detections = _get_node_attributes(masks)
    benchmark(_check_ctc, tracks, detections, masks)


def test_ctc_matched(benchmark, gt_data, pred_data):
    benchmark(CTCMatcher().compute_mapping, gt_data, pred_data)


@pytest.mark.timeout(300)
def test_ctc_metrics(benchmark, ctc_matched):
    def run_compute():
        return CTCMetrics().compute(copy.deepcopy(ctc_matched))

    ctc_results = benchmark.pedantic(run_compute, rounds=1, iterations=1)

    assert ctc_results["fn_edges"] == 87
    assert ctc_results["fn_nodes"] == 39
    assert ctc_results["fp_edges"] == 60
    assert ctc_results["fp_nodes"] == 0
    assert ctc_results["ns_nodes"] == 0
    assert ctc_results["ws_edges"] == 47


def test_ctc_div_metrics(benchmark, ctc_matched):
    def run_compute():
        return DivisionMetrics().compute(copy.deepcopy(ctc_matched))

    div_results = benchmark(run_compute)

    assert div_results["Frame Buffer 0"]["False Negative Divisions"] == 18
    assert div_results["Frame Buffer 0"]["False Positive Divisions"] == 30
    assert div_results["Frame Buffer 0"]["True Positive Divisions"] == 76


def test_iou_matched(benchmark, gt_data, pred_data):
    benchmark(IOUMatcher(iou_threshold=0.1).compute_mapping, gt_data, pred_data)


def test_iou_div_metrics(benchmark, iou_matched):
    def run_compute():
        return DivisionMetrics().compute(copy.deepcopy(iou_matched))

    div_results = benchmark(run_compute)

    assert div_results["Frame Buffer 0"]["False Negative Divisions"] == 25
    assert div_results["Frame Buffer 0"]["False Positive Divisions"] == 31
    assert div_results["Frame Buffer 0"]["True Positive Divisions"] == 69


# def test_track_overlap_metrics(benchmark, iou_matched):
#     def run_compute():
#         return TrackOverlapMetrics(copy.deepcopy(iou_matched)).compute()

#     benchmark(run_compute)
