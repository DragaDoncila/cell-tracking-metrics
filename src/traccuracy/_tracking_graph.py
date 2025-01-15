from __future__ import annotations

import copy
import enum
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Hashable, Iterable

import networkx as nx

if TYPE_CHECKING:
    import numpy as np
    from networkx.classes.reportviews import NodeView, OutEdgeView

logger = logging.getLogger(__name__)


@enum.unique
class NodeFlag(str, enum.Enum):
    """An enum containing standard flags that are used to annotate the nodes
    of a TrackingGraph. Note that the user specified frame and location
    attributes are also valid node attributes that will be stored on the graph
    and should not overlap with these values. Additionally, if a graph already
    has annotations using these strings before becoming a TrackingGraph,
    this will likely ruin metrics computation!
    """

    # True positive nodes as defined by CTC. Valid on gt and computed graphs.
    CTC_TRUE_POS = "is_ctc_tp"
    # False positive nodes as defined by CTC. Valid on computed graph.
    CTC_FALSE_POS = "is_ctc_fp"
    # False negative nodes as defined by CTC. Valid on gt graph.
    CTC_FALSE_NEG = "is_ctc_fn"
    # Non-split vertices as defined by CTC. Valid on computed graph
    # when many computed nodes can be matched to one gt node.
    NON_SPLIT = "is_ns"
    # True positive divisions. Valid on gt and computed graphs.
    TP_DIV = "is_tp_division"
    # False positive divisions. Valid on computed graph.
    FP_DIV = "is_fp_division"
    # False negative divisions. Valid on gt graph.
    FN_DIV = "is_fn_division"

    @classmethod
    def has_value(cls, value):
        """Check if a value is one of the enum's values.
        This can be used to check if other graph annotation strings are
        colliding with our reserved strings.

        Args:
            value : Check if the enum contains this value.

        Returns:
            bool: True if the value is already in the enum's values,
                false otherwise
        """
        return value in cls.__members__.values()


@enum.unique
class EdgeFlag(str, enum.Enum):
    """An enum containing standard flags that are used to
    annotate the edges of a TrackingGraph. If a graph already has
    annotations using these strings before becoming a TrackingGraph,
    this will likely ruin metrics computation!
    """

    # True positive edges. Valid on gt and computed graphs.
    TRUE_POS = "is_tp"
    # False positive edges as defined by CTC. Valid on computed graph.
    CTC_FALSE_POS = "is_ctc_fp"
    # False negative nodes as defined by CTC. Valid on gt graph.
    CTC_FALSE_NEG = "is_ctc_fn"
    # Edges between tracks as defined by CTC. Valid on gt and computed graphs.
    INTERTRACK_EDGE = "is_intertrack_edge"
    # Edges with wrong semantic as defined by CTC. Valid on computed graph.
    WRONG_SEMANTIC = "is_wrong_semantic"


class TrackingGraph:
    """A directed graph representing a tracking solution where edges go forward in time.

    Nodes represent cell detections and edges represent links between detections in the same track.
    Nodes in the graph must have an attribute that represents time frame (default to 't') and
    location (defaults to 'x' and 'y'). As in networkx, every cell must have a unique id, but these
    can be of any (hashable) type.

    We provide common functions for accessing parts of the track graph, for example
    all nodes in a certain frame, or all previous or next edges for a given node.
    Additional functionality can be accessed by querying the stored networkx graph
    with native networkx functions.

    Currently it is assumed that the structure of the networkx graph as well as the
    time frame and location of each node is not mutated after construction,
    although non-spatiotemporal attributes of nodes and edges can be modified freely.

    Attributes:
        start_frame: int, the first frame with a node in the graph
        end_frame: int, the end of the span of frames containing nodes
            (one frame after the last frame that contains a node)
        nodes_by_frame: dict of int -> node_id
            Maps from frames to all node ids in that frame
        frame_key: str
            The name of the node attribute that corresponds to the frame of
            the node. Defaults to "t".
        location_keys: list of str
            Keys used to access the location of the cell in space.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        segmentation: np.ndarray | None = None,
        frame_key: str = "t",
        label_key: str = "segmentation_id",
        location_keys: tuple[str, ...] = ("x", "y"),
        name: str | None = None,
    ):
        """A directed graph representing a tracking solution where edges go
        forward in time.

        If the provided graph already has annotations that are strings
        included in NodeFlags or EdgeFlags, this will likely ruin
        metric computation!

        Args:
            graph (networkx.DiGraph): A directed graph representing a tracking
                solution where edges go forward in time. If the graph already
                has annotations that are strings included in NodeFlags or
                EdgeFlags, this will likely ruin metrics computation!
            segmentation (numpy-like array, optional): A numpy-like array of segmentations.
                The location of each node in tracking_graph is assumed to be inside the
                area of the corresponding segmentation. Defaults to None.
            frame_key (str, optional): The key on each node in graph that
                contains the time frameof the node. Every node must have a
                value stored at this key. Defaults to 't'.
            label_key (str, optional): The key on each node that denotes the
                pixel value of the node in the segmentation. Defaults to
                'segmentation_id'. Pass `None` if there is not a label
                attribute on the graph.
            location_keys (tuple, optional): The list of keys on each node in
                graph that contains the spatial location of the node. Every
                node must have a value stored at each of these keys.
                Defaults to ('x', 'y').
            name (str, optional): User specified name that will be included in result
                outputs associated with this object
        """
        self.segmentation = segmentation
        if NodeFlag.has_value(frame_key):
            raise ValueError(
                f"Specified frame key {frame_key} is reserved for graph "
                "annotation. Please change the frame key."
            )
        self.frame_key = frame_key
        if label_key is not None and NodeFlag.has_value(label_key):
            raise ValueError(
                f"Specified label key {label_key} is reserved for graph"
                "annotation. Please change the label key."
            )
        self.label_key = label_key
        for loc_key in location_keys:
            if NodeFlag.has_value(loc_key):
                raise ValueError(
                    f"Specified location key {loc_key} is reserved for graph"
                    "annotation. Please change the location key."
                )
        self.location_keys = location_keys
        self.name = name

        self.graph = graph

        # construct dictionaries from attributes to nodes/edges for easy lookup
        self.nodes_by_frame: defaultdict[int, set[Hashable]] = defaultdict(set)
        self.nodes_by_flag: dict[NodeFlag, set[Hashable]] = {
            flag: set() for flag in NodeFlag
        }
        self.edges_by_flag: dict[EdgeFlag, set[tuple[Hashable, Hashable]]] = {
            flag: set() for flag in EdgeFlag
        }
        for node, attrs in self.graph.nodes.items():
            # check that every node has the time frame and location specified
            assert (
                self.frame_key in attrs.keys()
            ), f"Frame key {self.frame_key} not present for node {node}."
            for key in self.location_keys:
                assert (
                    key in attrs.keys()
                ), f"Location key {key} not present for node {node}."

            # store node id in nodes_by_frame mapping
            frame = attrs[self.frame_key]
            if frame not in self.nodes_by_frame.keys():
                self.nodes_by_frame[frame] = {node}
            else:
                self.nodes_by_frame[frame].add(node)
            # store node id in nodes_by_flag mapping
            for node_flag in NodeFlag:
                if attrs.get(node_flag):
                    self.nodes_by_flag[node_flag].add(node)

        # store edge id in edges_by_flag
        for edge, attrs in self.graph.edges.items():
            for edge_flag in EdgeFlag:
                if attrs.get(edge_flag):
                    self.edges_by_flag[edge_flag].add(edge)

        # Store first and last frames for reference
        if len(self.nodes_by_frame) == 0:
            self.start_frame = None
            self.end_frame = None
        else:
            self.start_frame = min(self.nodes_by_frame.keys())
            self.end_frame = max(self.nodes_by_frame.keys()) + 1

        # Record types of annotations that have been calculated
        self.division_annotations = False
        self.node_errors = False
        self.edge_errors = False

    @property
    def nodes(self) -> NodeView:
        """Get all the nodes in the graph, along with their attributes.

        Returns:
            NodeView: Provides set-like operations on the nodes as well as node attribute lookup.
        """
        return self.graph.nodes

    @property
    def edges(self) -> OutEdgeView:
        """Get all the edges in the graph, along with their attributes.

        Returns:
            OutEdgeView: Provides set-like operations on the edge-tuples as well as edge attribute
                lookup.
        """
        return self.graph.edges

    def get_location(self, node_id: Hashable) -> list[float]:
        """Get the spatial location of the node with node_id using self.location_keys.

        Args:
            node_id (hashable): The node_id to get the location of

        Returns:
            list of float: A list of location values in the same order as self.location_keys
        """
        return [self.graph.nodes[node_id][key] for key in self.location_keys]

    def get_nodes_with_flag(self, flag: NodeFlag) -> set[Hashable]:
        """Get all nodes with specified NodeFlag set to True.

        Args:
            flag (traccuracy.NodeFlag): the node flag to query for

        Returns:
            (List(hashable)): An iterable of node_ids which have the given flag
                and the value is True.
        """
        if not isinstance(flag, NodeFlag):
            raise ValueError(f"Function takes NodeFlag arguments, not {type(flag)}.")
        return self.nodes_by_flag[flag]

    def get_edges_with_flag(self, flag: EdgeFlag) -> set[tuple[Hashable, Hashable]]:
        """Get all edges with specified EdgeFlag set to True.

        Args:
            flag (traccuracy.EdgeFlag): the edge flag to query for

        Returns:
            (List(hashable)): An iterable of edge ids which have the given flag
                and the value is True.
        """
        if not isinstance(flag, EdgeFlag):
            raise ValueError(f"Function takes EdgeFlag arguments, not {type(flag)}.")
        return self.edges_by_flag[flag]

    def get_divisions(self) -> list[Hashable]:
        """Get all nodes that have at least two edges pointing to the next time frame

        Returns:
            list of hashable: a list of node ids for nodes that have more than one child
        """
        return [node for node, degree in self.graph.out_degree() if degree >= 2]

    def get_merges(self) -> list[Hashable]:
        """Get all nodes that have at least two incoming edges from the previous time frame

        Returns:
            list of hashable: a list of node ids for nodes that have more than one parent
        """
        return [node for node, degree in self.graph.in_degree() if degree >= 2]

    def get_connected_components(self) -> list[TrackingGraph]:
        """Get a list of TrackingGraphs, each corresponding to one track
        (i.e., a connected component in the track graph).

        Returns:
            A list of TrackingGraphs, one for each track.
        """
        graph = self.graph
        if len(graph.nodes) == 0:
            return []

        return [self.get_subgraph(g) for g in nx.weakly_connected_components(graph)]

    def get_subgraph(self, nodes: Iterable[Hashable]) -> TrackingGraph:
        """Returns a new TrackingGraph with the subgraph defined by the list of nodes.

        Args:
            nodes (list): A list of node ids to use in constructing the subgraph
        """
        new_graph = self.graph.subgraph(nodes).copy()

        new_trackgraph = copy.deepcopy(self)
        new_trackgraph.graph = new_graph
        for frame, nodes_in_frame in self.nodes_by_frame.items():
            new_nodes_in_frame = nodes_in_frame.intersection(nodes)
            if new_nodes_in_frame:
                new_trackgraph.nodes_by_frame[frame] = new_nodes_in_frame
            else:
                del new_trackgraph.nodes_by_frame[frame]

        for node_flag in NodeFlag:
            new_trackgraph.nodes_by_flag[node_flag] = self.nodes_by_flag[
                node_flag
            ].intersection(nodes)
        for edge_flag in EdgeFlag:
            new_trackgraph.edges_by_flag[edge_flag] = self.edges_by_flag[
                edge_flag
            ].intersection(new_trackgraph.edges)

        if len(new_trackgraph.nodes_by_frame) == 0:
            new_trackgraph.start_frame = None
            new_trackgraph.end_frame = None
        else:
            new_trackgraph.start_frame = min(new_trackgraph.nodes_by_frame.keys())
            new_trackgraph.end_frame = max(new_trackgraph.nodes_by_frame.keys()) + 1

        return new_trackgraph

    def set_flag_on_node(
        self, _id: Hashable, flag: NodeFlag, value: bool = True
    ) -> None:
        """Set an attribute flag for a single node.
        If the id is not found in the graph, a KeyError will be raised.
        If the flag already exists, the existing value will be overwritten.

        Args:
            _id (Hashable): The node id on which to set the flag.
            flag (traccuracy.NodeFlag): The node flag to set. Must be
                of type NodeFlag - you may not not pass strings, even if they
                are included in the NodeFlag enum values.
            value (bool, optional): Flags can only be set to
                True or False. Defaults to True.

        Raises:
            KeyError if the provided id is not in the graph.
            ValueError if the provided flag is not a NodeFlag
        """
        if not isinstance(flag, NodeFlag):
            raise ValueError(
                f"Provided  flag {flag} is not of type NodeFlag. "
                "Please use the enum instead of passing string values."
            )
        self.graph.nodes[_id][flag] = value
        if value:
            self.nodes_by_flag[flag].add(_id)
        else:
            self.nodes_by_flag[flag].discard(_id)

    def remove_flag_from_node(self, _id: Hashable, flag: NodeFlag) -> None:
        """Removes a flag from a node

        Args:
            _id (Hashable): The node id on which to set the flag.
            flag (NodeFlag): The node flag to set. Must be
                of type NodeFlag - you may not not pass strings, even if they
                are included in the NodeFlag enum values.

        Raises:
            KeyError if the flag is not present on the node.
        """

        if flag not in self.graph.nodes[_id]:
            raise KeyError(f"Provided {flag} is not present on node {_id}.")

        del self.graph.nodes[_id][flag]
        self.nodes_by_flag[flag].discard(_id)

    def set_flag_on_all_nodes(self, flag: NodeFlag, value: bool = True) -> None:
        """Set an attribute flag for all nodes in the graph.
        If the flag already exists, the existing values will be overwritten.

        Args:
            flag (traccuracy.NodeFlag): The node flag to set. Must be
                of type NodeFlag - you may not not pass strings, even if they
                are included in the NodeFlag enum values.
            value (bool, optional): Flags can only be set to True or False.
                Defaults to True.

        Raises:
            ValueError if the provided flag is not a NodeFlag.
        """
        if not isinstance(flag, NodeFlag):
            raise ValueError(
                f"Provided  flag {flag} is not of type NodeFlag. "
                "Please use the enum instead of passing string values."
            )
        nx.set_node_attributes(self.graph, value, name=flag)
        if value:
            self.nodes_by_flag[flag].update(self.graph.nodes)
        else:
            self.nodes_by_flag[flag] = set()

    def set_flag_on_edge(
        self, _id: tuple[Hashable, Hashable], flag: EdgeFlag, value: bool = True
    ) -> None:
        """Set an attribute flag for an edge.
        If the flag already exists, the existing value will be overwritten.

        Args:
            ids (tuple[Hashable]): The edge id or list of edge ids
                to set the attribute for. Edge ids are a 2-tuple of node ids.
            flag (traccuracy.EdgeFlag): The edge flag to set. Must be
                of type EdgeFlag - you may not pass strings, even if they are
                included in the EdgeFlag enum values.
            value (bool): Flags can only be set to True or False.
                Defaults to True.

        Raises:
            KeyError if edge with _id not in graph.
        """
        if not isinstance(flag, EdgeFlag):
            raise ValueError(
                f"Provided attribute {flag} is not of type EdgeFlag. "
                "Please use the enum instead of passing string values."
            )
        self.graph.edges[_id][flag] = value
        if value:
            self.edges_by_flag[flag].add(_id)
        else:
            self.edges_by_flag[flag].discard(_id)

    def remove_flag_from_edge(
        self, _id: tuple[Hashable, Hashable], flag: EdgeFlag
    ) -> None:
        """Removes flag from a given node

        Args:
            ids (tuple[Hashable]): The edge id or list of edge ids
                to set the attribute for. Edge ids are a 2-tuple of node ids.
            flag (traccuracy.EdgeFlag): The edge flag to set. Must be
                of type EdgeFlag - you may not pass strings, even if they are
                included in the EdgeFlag enum values.

        Raises:
            KeyError if edge with _id not in graph.
            KeyError if flag not present on edge
        """

        if flag not in self.graph.edges[_id]:
            raise KeyError(f"Flag {flag} not present on edge {_id}.")

        del self.graph.edges[_id][flag]
        self.edges_by_flag[flag].discard(_id)

    def set_flag_on_all_edges(self, flag: EdgeFlag, value: bool = True) -> None:
        """Set an attribute flag for all edges in the graph.
        If the flag already exists, the existing values will be overwritten.

        Args:
            flag (traccuracy.EdgeFlag): The edge flag to set. Must be
                of type EdgeFlag - you may not not pass strings, even if they
                are included in the EdgeFlag enum values.
            value (bool, optional): Flags can only be set to True or False.
                Defaults to True.

        Raises:
            ValueError if the provided flag is not an EdgeFlag.
        """
        if not isinstance(flag, EdgeFlag):
            raise ValueError(
                f"Provided  flag {flag} is not of type EdgeFlag. "
                "Please use the enum instead of passing string values, "
                "and add new attributes to the class to avoid key collision."
            )
        nx.set_edge_attributes(self.graph, value, name=flag)
        if value:
            self.edges_by_flag[flag].update(self.graph.edges)
        else:
            self.edges_by_flag[flag] = set()

    def get_tracklets(
        self, include_division_edges: bool = False
    ) -> list[TrackingGraph]:
        """Gets a list of new TrackingGraph objects containing all tracklets of the current graph.

        Tracklet is defined as all connected components between divisions (daughter to next
        parent). Tracklets can also start or end with a non-dividing cell.

        Args:
            include_division_edges (bool, optional): If True, include edges at division.

        """

        graph_copy = self.graph.copy()

        # Remove all intertrack edges from a copy of the original graph
        removed_edges = []
        for parent in self.get_divisions():
            for daughter in self.graph.successors(parent):
                graph_copy.remove_edge(parent, daughter)
                removed_edges.append((parent, daughter))

        # Extract subgraphs (aka tracklets) and return as new track graphs
        tracklets = nx.weakly_connected_components(graph_copy)

        if include_division_edges:
            tracklets = list(tracklets)
            # Add back intertrack edges
            for tracklet in tracklets:
                for parent, daughter in removed_edges:
                    if daughter in tracklet:
                        tracklet.add(parent)

        return [self.get_subgraph(g) for g in tracklets]
