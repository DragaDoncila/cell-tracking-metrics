Matchers
========

The first step in comparing a ground truth and predicted graph is computing a matching
between the two graphs. Currently, traccuracy represents a matching between graphs
as a matching between the nodes of the graphs - edges are considered matched if the
endpoints of the edge are matched. Traccuracy supports three types of matchings, depending
on how many nodes in one graph can be matched to one node in the other graph: 
* many-to-one: many ground truth nodes may be matched to one predicted node
* one-to-many: one ground truth node may be matched to many predicted nodes
* one-to-one: every node can be matched to at most one node in the other graph

Many metrics only operate on one-to-one matchings, or support at most one of the "many"
mappings. Matchers can operate on the segmentation or directly on the point locations.
Below is a table summarizing the implemented matchers, what types of matchings
they can produce, and a brief description of behavior and any hyperparameters.

.. list-table:: Implemented Matchers
    :widths: auto
    :header-rows: 1

    * - Matcher
      - Matching Type(s)
      - Description
    * - Point Matcher
      - one-to-one
      - Given a maximum distance threshold, the matcher will perform hungarian matching on the points in each
        frame of the ground truth and predicted graphs, minimizing the overall distance while
        never matching any points with distance greater than the threshold.
    * - IOU Matcher
      - one-to-one, many-to-one, one-to-many
      - Given a minimum overlap threshold, will match the segmentations
        in each frame of the ground truth and predicted tracks with 
        intersection-over-union greater than or equal to the given threshold.
        If the one-to-one flag is true, will produce a one-to-one matching by running
        linear assignment/hungarian matching on the thresholded iou array.
    * - CTC Matcher
      - one-to-one, many-to-one
      - A predicted node is matched to a reference node if the computed
        segmentation covers a majority of the reference segmentation.
        See https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144959
        for complete details.