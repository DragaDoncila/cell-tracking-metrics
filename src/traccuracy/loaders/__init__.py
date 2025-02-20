"""Subpackage for loading tracking data into memory

This subpackage contains functions for loading ground
truth or tracking method outputs into memory as TrackingGraph objects.
Each loading function must return one TrackingGraph object which has a
track graph and optionally contains a corresponding segmentation.
"""

from ._ctc import _check_ctc, _get_node_attributes, _load_tiffs, load_ctc_data

__all__ = ["_check_ctc", "_get_node_attributes", "_load_tiffs", "load_ctc_data"]
