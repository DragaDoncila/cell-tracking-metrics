Gap-Closing Edges
=================

Gap-closing edges are those which connect a node in frame ``t`` to a node in frame ``t+2`` onwards. They are also called frame-skip edges.

Traccuracy allows for gap-closing edges in both the ground truth and predicted graphs. For the purposes of error classification,
an edge ``u -> v`` is **not** considered identical to a matching edge ``u -> w -> v``, and this will lead to errors annotated in this region.
The specific errors annotated will depend on the type of errors requested, and they are detailed in the other Track Error sections.