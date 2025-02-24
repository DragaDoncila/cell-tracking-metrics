# traccuracy: Evaluate Cell Tracking Solutions

[![License](https://img.shields.io/pypi/l/traccuracy.svg?color=green)](https://github.com/Janelia-Trackathon-2023/traccuracy/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/traccuracy.svg?color=green)](https://pypi.org/project/traccuracy)
[![Python Version](https://img.shields.io/pypi/pyversions/traccuracy.svg?color=green)](https://python.org)
[![CI](https://github.com/Janelia-Trackathon-2023/traccuracy/actions/workflows/ci.yml/badge.svg)](https://github.com/Janelia-Trackathon-2023/traccuracy/actions/workflows/ci.yml)
[![Benchmarking](https://github.com/Janelia-Trackathon-2023/traccuracy/actions/workflows/benchmark-report.yml/badge.svg)](https://janelia-trackathon-2023.github.io/traccuracy/dev/bench/)
[![Documentation Status](https://readthedocs.org/projects/traccuracy/badge/?version=latest)](https://traccuracy.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Janelia-Trackathon-2023/traccuracy/branch/main/graph/badge.svg)](https://codecov.io/gh/Janelia-Trackathon-2023/traccuracy)


`traccuracy` provides a suite of benchmarking functions that can be used to evaluate cell
tracking solutions against ground truth annotations. The goal of this library is to provide a convenient way to run rigorous evaluation and to document and consolidate the wide variety of
metrics used in the field.

`traccuracy` can compute a comprehensive set of metrics for evaluating the cell linking and division performance, and can compute biologically meaningful metrics such as number of correctly reconstructed lineages over N frames and cell cycle length accuracy. As matching ground truth and
predicted lineages is a crucial step for performing evaluation, `traccuracy` includes
a number of algorithms for matching ground truth and predicted lineages, both with
and without segmentation masks.

Learn more in the [documentation](https://traccuracy.readthedocs.io/en/latest/) or check out the [source code](https://github.com/Janelia-Trackathon-2023/traccuracy).

## Installation
`pip install traccuracy`

## Getting Started
The `traccuracy` library has three main components: loaders, matchers, and metrics.
Loaders load tracking graphs from other formats, such as the CTC format, into a [TrackingGraph](https://traccuracy.readthedocs.io/en/latest/autoapi/traccuracy/index.html#traccuracy.TrackingGraph) object.
A TrackingGraph is a spatiotemporal graph.
Nodes represent a single cell in a given time point, and are annotated with a time and a location.
Edges point forward in time from a node representing a cell in time point `t` to the same cell or its daughter in frame `t+1` (or beyond, to represent gap-closing).
To load TrackingGraphs from a custom format, you will likely need to implement a loader: see
documentation [here](https://traccuracy.readthedocs.io/en/latest/autoapi/traccuracy/loaders/index.html#module-traccuracy.loaders) for more information.

Matchers take a ground truth and a predicted TrackingGraph with optional segmentation masks and match the nodes and edges to allow evaluation to occur.
Metrics are then computed on the matched graphs, and a summary is printed out.

The `traccuracy` library has a command line interface for running common metrics
pipelines, [documented here](https://traccuracy.readthedocs.io/en/latest/cli.html), and a flexible Python API, shown in [this](https://traccuracy.readthedocs.io/en/latest/examples/ctc.html) example notebook.


## Implemented Metrics

 - CTC-DET from the [Cell Tracking Challenge Evaluation Methodology](http://celltrackingchallenge.net/evaluation-methodology/)
 - CTC-TRA from the [Cell Tracking Challenge Evaluation Methodology](http://celltrackingchallenge.net/evaluation-methodology/)
 - Acyclic Oriented Graph Metric (AOGM) from [Matula et al. 2015](https://doi.org/10.1371/journal.pone.0144959). A generalized form the CTC metrics where you can supply different weights for each component of the overall metric.
 - Division Precision. Optionally allows detection within N frames of ground truth division.
 - Division Recall. Optionally allows detection within N frames of ground truth division.
 - Division F1 score. Optionally allows detection within N frames of ground truth division.
 - Mitotic Branching Correctness from [Ulicna et al. 2021](https://doi.org/10.3389/fcomp.2021.734559). TP / (TP + FP + FN). Optionally allows detection within N frames of ground truth division.

## Glossary

**Tracklet**
: A single non-dividing cell tracked over time. In graph terms, this is the connected component of a track between divisions (daughter to next parent). Tracklets can also start or end with a non-dividing cell at the beginning and end of the captured time or if the track leaves the field of view.

**Track**
: A single cell and all of its progeny. In graph terms, a connected component including divisions.

**Gap-Closing**
: Also known as *frame-skipping*, these are edges that connect non-consecutive frames to signify a cell being occluded or missing for some frames, before the track continues.