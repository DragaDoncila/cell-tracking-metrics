from typing import Any

import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops


def make_one_cell_2d(
    label: int = 1,
    arr_shape: tuple[int, int] = (32, 32),
    center: tuple[int, int] = (16, 16),
    radius: int = 7,
) -> np.ndarray:
    """Create a 2D numpy array with a single circular cell.

    Args:
        label (int, optional): Value of mask in the foreground. Defaults to 1.
        arr_shape (tuple[int, int], optional): The size of the numpy array to return.
            Defaults to (32, 32).
        center (tuple[int, int], optional): The center of the cell, in pixels.
            Defaults to (16, 16).
        radius (int, optional): The radius of the cell. Defaults to 7.

    Returns:
        np.array: A numpy array with a circle at the given center with the
            given label.
    """
    im = np.zeros(arr_shape, dtype="int32")
    rr, cc = disk(center, radius, shape=arr_shape)
    im[rr, cc] = label
    return im


def make_split_cell_2d(
    labels=(1, 2), arr_shape=(32, 32), center=(16, 16), radius=9
) -> np.ndarray:
    """Create a 2d numpy array with two cells, each half a circle.

    Args:
        labels (tuple, optional): _description_. Defaults to (1, 2).
        arr_shape (tuple, optional): _description_. Defaults to (32, 32).
        center (tuple, optional): _description_. Defaults to (16, 16).
        radius (int, optional): _description_. Defaults to 7.

    Returns:
        np.ndarray : A numpy array with two half circles with the given labels.
            The pixels with y value greater than center will be the second label
            color, and those with y value lass than or equal to the center
            will have the first label.
    """
    im = np.zeros(arr_shape, dtype="int32")
    rr, cc = disk(center, radius, shape=arr_shape)
    im[rr, cc] = labels[0]
    # get indices where y value greater than center
    mask = cc > center[1]
    im[rr[mask], cc[mask]] = labels[1]
    return im


def sphere(
    center: tuple[int, int, int], radius: int, shape: tuple[int, int, int]
) -> np.ndarray:
    """Get a mask of a sphere of a given radius

    Args:
        center (tuple[int, int, int]): The coordinate of the center of the sphere.
        radius (int): The radius of the sphere
        shape (tuple[int, int, int]): The share of the numpy array mask to return.

    Returns:
        np.ndarray: A boolean array with 1s inside the sphere and 0s outside.
    """
    assert len(center) == len(shape)
    indices = np.moveaxis(np.indices(shape), 0, -1)  # last dim is the index
    distance = np.linalg.norm(np.subtract(indices, np.asarray(center)), axis=-1)
    mask = distance <= radius
    return mask


def make_one_cell_3d(
    label=1, arr_shape=(32, 32, 32), center=(16, 16, 16), radius=7
) -> np.ndarray:
    """Make a numpy array containing a single (spherical) cell in 3d.

    Args:
        label (int, optional): _description_. Defaults to 1.
        arr_shape (tuple, optional): _description_. Defaults to (32, 32, 32).
        center (tuple, optional): _description_. Defaults to (16, 16, 16).
        radius (int, optional): _description_. Defaults to 7.

    Returns:
        np.ndarray: A numpy array of the given shape containing a sphere
            with the given label, radius, and center.

    """
    im = np.zeros(arr_shape, dtype="int32")
    mask = sphere(center, radius, shape=arr_shape)
    im[mask] = label
    return im


def make_split_cell_3d(
    labels=(1, 2), arr_shape=(32, 32, 32), center=(16, 16, 16), radius=9
):
    """Make a numpy array containing two cells, each half a sphere.
    The pixels with y value less than or equal to the center y value will have
    the first label, and those with y value greater than the center will
    have the second label

    Args:
        labels (tuple, optional): _description_. Defaults to (1, 2).
        arr_shape (tuple, optional): _description_. Defaults to (32, 32, 32).
        center (tuple, optional): _description_. Defaults to (16, 16, 16).
        radius (int, optional): _description_. Defaults to 9.

    Returns:
        np.ndarray: A numpy array of the given shape containing a sphere
            with the given radius, and center. Half the sphere has the first,
            label, and the other half has the second label.
    """
    im = np.zeros(arr_shape, dtype="int32")
    mask = sphere(center, radius, shape=arr_shape)
    im[mask] = labels[0]
    # get indices where y value greater than center
    mask[:, 0 : center[1] + 1] = 0
    im[mask] = labels[1]
    return im


### CANONICAL 2D SEGMENTATION EXAMPLES ###
def good_segmentation_2d() -> tuple[np.ndarray, np.ndarray]:
    """A pretty good (but not perfect) pair of segmentations in 2d.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations of
            a single cell. The segmentations are circles of the same size with
            a slight offset in x and y.
    """
    gt = make_one_cell_2d(label=1, center=(15, 15), radius=9)
    pred = make_one_cell_2d(label=2, center=(17, 17), radius=9)
    return gt, pred


def false_positive_segmentation_2d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt is empty and the prediction has a
    single cell.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations of
            a single cell. The gt is empty and the prediction has a single cell.
    """
    gt = np.zeros((32, 32), dtype="int32")
    pred = make_one_cell_2d(label=1, center=(17, 17), radius=9)
    return gt, pred


def false_negative_segmentation_2d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt has a single cell and the
    prediction is empty.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations of
            a single cell. The pred is empty and the gt has a single cell.
    """
    gt = make_one_cell_2d(label=1, center=(15, 15), radius=9)
    pred = np.zeros((32, 32), dtype="int32")
    return gt, pred


def oversegmentation_2d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt has a single cell and the prediction
    splits that into two cells.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations.
        The gt has a single circle labeled and the pred splits that circle
        into two labels.
    """
    gt = make_one_cell_2d(label=1, center=(16, 16), radius=9)
    pred = make_split_cell_2d(labels=(2, 3), center=(16, 16), radius=9)
    return gt, pred


def undersegmentation_2d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt has two cells and the prediction
    merges them into one circular cell.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations.
        The pred has a single merged circle labeled and the gt has two labels,
        each half of the circle.
    """
    gt = make_split_cell_2d(labels=(1, 2), center=(16, 16), radius=9)
    pred = make_one_cell_2d(label=3, center=(16, 16), radius=9)
    return gt, pred


def no_overlap_2d() -> tuple[np.ndarray, np.ndarray]:
    """Two cells with no overlap in 2d."""
    gt = make_one_cell_2d(label=1, center=(5, 5), radius=7)
    pred = make_one_cell_2d(label=2, center=(17, 17), radius=7)
    return gt, pred


def multicell_2d() -> tuple[np.ndarray, np.ndarray]:
    """Two cells in each image, one that overlaps and one that doesn't"""
    arr_shape = (32, 32)
    radius = 5

    gt = np.zeros(arr_shape, dtype="int32")
    pred = np.zeros(arr_shape, dtype="int32")

    # Overlap cell
    rr, cc = disk((5, 5), radius, shape=arr_shape)
    gt[rr, cc] = 1
    pred[rr, cc] = 3

    # Unique gt
    rr, cc = disk((17, 17), radius, shape=arr_shape)
    gt[rr, cc] = 2

    # Unique pred
    rr, cc = disk((25, 7), radius, shape=arr_shape)
    pred[rr, cc] = 4

    return gt, pred


### CANONICAL 3D SEGMENTATION EXAMPLES ###
def good_segmentation_3d() -> tuple[np.ndarray, np.ndarray]:
    """A pretty good (but not perfect) pair of segmentations in 3d.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations of
            a single cell. The segmentations are circles of the same size with
            a slight offset in x and y.
    """
    gt = make_one_cell_3d(label=1, center=(15, 15, 15), radius=9)
    pred = make_one_cell_3d(label=2, center=(17, 17, 17), radius=9)
    return gt, pred


def false_positive_segmentation_3d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt is empty and the prediction has a
    single cell.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations of
            a single cell. The gt is empty and the prediction has a single cell.
    """
    gt = np.zeros((32, 32, 32), dtype="int32")
    pred = make_one_cell_3d(label=1, center=(17, 17, 17), radius=9)
    return gt, pred


def false_negative_segmentation_3d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt has a single cell and the
    prediction is empty.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations of
            a single cell. The pred is empty and the gt has a single cell.
    """
    gt = make_one_cell_3d(label=1, center=(15, 15, 15), radius=9)
    pred = np.zeros((32, 32), dtype="int32")
    return gt, pred


def oversegmentation_3d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt has a single cell and the prediction
    splits that into two cells.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations.
        The gt has a single circle labeled and the pred splits that circle
        into two labels.
    """
    gt = make_one_cell_3d(label=1, center=(16, 16, 16), radius=9)
    pred = make_split_cell_3d(labels=(2, 3), center=(16, 16, 16), radius=9)
    return gt, pred


def undersegmentation_3d() -> tuple[np.ndarray, np.ndarray]:
    """A pair of segmentations where the gt has two cells and the prediction
    merges them into one circular cell.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of (gt, pred) segmentations.
        The pred has a single merged circle labeled and the gt has two labels,
        each half of the circle.
    """
    gt = make_split_cell_3d(labels=(1, 2), center=(16, 16, 16), radius=9)
    pred = make_one_cell_3d(label=3, center=(16, 16, 16), radius=9)
    return gt, pred


def no_overlap_3d() -> tuple[np.ndarray, np.ndarray]:
    """3D segmentations with no overlap"""
    gt = make_one_cell_3d(label=1, center=(5, 5, 5), radius=5)
    pred = make_one_cell_3d(label=2, center=(17, 17, 17), radius=6)
    return gt, pred


def multicell_3d() -> tuple[np.ndarray, np.ndarray]:
    """Two cells in each image, one that overlaps and one that doesn't"""
    arr_shape = (32, 32, 32)
    radius = 5

    gt = np.zeros(arr_shape, dtype="int32")
    pred = np.zeros(arr_shape, dtype="int32")

    # Overlap cell
    mask = sphere((5, 5, 5), radius, shape=arr_shape)
    gt[mask] = 1
    pred[mask] = 3

    # Unique gt
    mask = sphere((17, 17, 17), radius, shape=arr_shape)
    gt[mask] = 2

    # Unique pred
    mask = sphere((25, 7, 7), radius, shape=arr_shape)
    pred[mask] = 4

    return gt, pred


def nodes_from_segmentation(
    seg: np.ndarray,
    frame: int = 0,
    pos_keys=("y", "x"),
    frame_key="t",
    label_key="segmentation_id",
    _id="label",
) -> dict[Any, dict]:
    """Extract candidate nodes from a segmentation. Also computes specified attributes.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels, representing one time
            frame.
        frame (int, optional): The time frame of this array. Used for making node ids and
            for populating the attributes dict. Defaults to 0
        pos_keys (tuple[str], optional): The attribute keys to use to store the positions.
            Defaults to ("y", "x")
        frame_key (str, optional): The frame key to use in the attributes dict.
            Defaults to "t".
        label_key (str, optional): The label key to use in the attributes dict.
            Defaults to "label_id"
        _id (str, optional): What to use for node ids. Options are: "label" - the label
            value as an integer, "label_time" - a string with format f"{label}_{time}"

    Returns:
        dict[Any, dict]: A dictionary from node_ids to node attributes, which
            can be used to create a networkx graph using add_nodes_from().
            Node Ids are currently label_id. Attributes include the
            frame and the label.
    """
    nodes = {}
    props = regionprops(seg)
    for regionprop in props:
        if _id == "label":
            node_id = regionprop.label
        elif _id == "label_time":
            node_id = f"{regionprop.label}_{frame}"
        attrs = {frame_key: frame, label_key: regionprop.label}
        centroid = regionprop.centroid
        assert len(pos_keys) == len(centroid), (
            f"Number of position keys {pos_keys} does not match number of "
            f"elements in centroid {centroid}"
        )
        for key, val in zip(pos_keys, centroid):
            attrs[key] = val
        nodes[node_id] = attrs
    return nodes
