---
title: "PyGraphon: a Python package for graphon estimation"
tags:
  - Python
  - statistics
  - graph limit
  - graphon
  - networks
authors:
  - name: Charles Dufour
    orcid: 0009-0000-3612-7335
    equal-contrib: true
    affiliation: 1
  - name: Arthur Verdeyme
    orcid: 0009-0003-5319-9228
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: EPFL, Switzerland
    index: 1
date: 12 December 2023
bibliography: paper.bib
---

# Summary

_describing the high-level functionality and purpose of the software for a
diverse, non-specialist audience_

`Pygraphon` is a Python library designed to estimate graphons, providing an
intuitive interface for researchers and practitioners to model and analyze
graph structures efficiently. Graphons are probability distributions on the
space of graphs, allowing users to infer underlying network characteristics
from observed data. This library caters to a broad audience, enabling
non-specialists to engage with graphon estimation techniques and apply them in
various fields, including social networks, neuroscience, and machine learning.

# Statement of need

_clearly illustrates the research purpose of the software and places it in the
context of related work._

Graphon estimation stands as a crucial tool in the analysis of large-scale
networks, serving as a fundamental step in understanding complex systems.
`Pygraphon` addresses the need for accessible, user-friendly tools that enable
researchers across domains to estimate graphons accurately. By providing a
comprehensive set of functionalities for graphon modeling, estimation, and
visualization, this library contributes to the advancement of network science
and facilitates insights into the structure and dynamics of complex networks.

## Similar software

_A list of key references, including to other software addressing related
needs. Note that the references should include full names of venues, e.g.,
journals and conferences, not abbreviations only understood in the context of a
specific discipline._

- `R` package [`graphon`](https://cran.r-project.org/package=graphon)
  [@graphon]
- `R` package [`nethist`](https://enigmasong.github.io/nethist/index.html)
  _(development stage)_ [@song2023]
- `Julia` package
  [`NetworkHistogram.jl`](https://docs.juliahub.com/General/NetworkHistogram/stable/)
  [@Dufour2023]

We list the different methods implemented in `Pygraphon`, `graphon`, and
`NetworkHistogram.jl` latter in Table 2.

_Mention (if applicable) a representative set of past or ongoing research
projects using the software and recent scholarly publications enabled by it._

Recent Scholarly Publications Enabled by `Pygraphon`:

- @verdeyme2024shape

# Functionality and implementation

For more details, please refer to the [documentation](https://pygraphon.readthedocs.io/en/latest/).

$$ A\_{ij} | \xi_i, \xi_j \sim \mathrm{Bernoulli}(W(\xi_i, \xi_j)) $$

| Methods                | class names          | Reference            |
| ---------------------- | -------------------- | -------------------- |
| Matrix completion      | `Completion`         | [@Keshavan2010]      |
| Largest Gap algorithm  | `LG`                 | [@channarond2012]    |
| Sort and Smooth        | `SAS`                | [@chan2014]          |
| Network Histogram      | `HistogramEstimator` | [@olhede2014]        |
| Neighborhood smoothing | `NBDsmooth`          | [@Zhang2015]         |
| USVT                   | `USVT`               | [@Chatterjee2015]    |
| Shape Histogram        | `SmoothNetHist`      | [@verdeyme2024shape] |

**Table 1:** Comparison of different methods implemented in `Pygraphon` with
their corresponding class names and references.

## Comparison with existing implementations

`nethist` and `NetworkHistogram.jl` only implement the network histogram
method.

|                        | `Pygraphon` | `graphon` |
| :--------------------- | :---------: | :-------: |
| Matrix completion      |     Yes     |    Yes    |
| Largest Gap            |     Yes     |    Yes    |
| SBA[^1]                |             |    Yes    |
| Sort and Smooth        |     Yes     |    Yes    |
| Network Histogram      |     Yes     |    Yes    |
| Neighborhood smoothing |     Yes     |    Yes    |
| USVT                   |     Yes     |    Yes    |
| Shape Histogram        |     Yes     |           |

**Table 2:** Comparison of different methods implemented in `Pygraphon`,
`graphon`, and `NetworkHistogram.jl` with their availability indicated by
"Yes".

# References

[^1]:
    the SBA algorithm [@Airoldi2013] is not implemented in `Pygraphon` yet as
    it is intended to be used with multiple graphs and not a single observed
    graph. We plan to implement it in the future.
