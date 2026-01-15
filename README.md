<!--
<p align="center">
	<img src="https://github.com/dufourc1/pygraphon/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
	PyGraphon
</h1>


<p>
        <a href="https://zenodo.org/doi/10.5281/zenodo.10355247"><img src="https://zenodo.org/badge/446324463.svg" alt="DOI"></a>
        <a href='https://pygraphon.readthedocs.io/en/latest/index.html'>
				<img src='https://readthedocs.org/projects/pygraphon/badge/?version=latest' alt='Documentation Status' />
		</a>
        <a href="https://codecov.io/gh/dufourc1/pygraphon" >
            <img src="https://codecov.io/gh/dufourc1/pygraphon/branch/master/graph/badge.svg?token=MDWJ6F86US"/>
        </a>
</p>
<p>
        <a href="https://github.com/dufourc1/pygraphon/actions/workflows/build.yml">
            <img alt="Build" src="https://github.com/dufourc1/pygraphon/workflows/build/badge.svg" />
        </a>
		<a href="https://pypi.org/project/pygraphon">
				<img alt="PyPI" src="https://img.shields.io/pypi/v/pygraphon" />
		</a>
		<a href="https://pypi.org/project/pygraphon">
				<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/pygraphon" />
		</a>
		<a href='https://github.com/psf/black'>
				<img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
		</a>
</p>

Welcome to the documentation of our Python library for estimating graphons from observed data! Our library provides a powerful set of tools for estimating graphons, which are a type of object used to model large, random graphs, from observed data.

```python
from pygraphon import HistogramEstimator

#  sample a graph with n nodes from one of the built-in graphons f(x,y) = x*y
from pygraphon import graphon_graphon_product
A = graphon_product.draw(n = 100)

# Create a histogram estimator
estimator = HistogramEstimator()

# Fit the estimator to a graph with adjacency matrix A
estimator.fit(A)

# Get the estimated graphon
graphon_estimated = estimator.get_graphon()

# get the estimated block connectivity matrix
theta = graphon_estimated.get_theta()

# get the estimated edge probability matrix
P_estimated = graphon_estimated.get_edge_connectivity()
```


To cite this library, please use the following citation:

```bibtex
@misc{dufour_verdeyme_2023,
         author       = {Dufour, Charles and Verdeyme, Arthur},
         title        = {PyGraphon},
         year         = {2023},
         howpublished = {https://doi.org/10.5281/zenodo.10355247},
         note         = {Zenodo},
         doi          = {10.5281/zenodo.10355247}
    }
```

## 🚀 Installation

<details>
 <summary>See Installation instructions</summary>
The most recent release can be installed from
[PyPI](https://pypi.org/project/pygraphon/) with:

```bash
$ pip install pygraphon
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/dufourc1/pygraphon.git
```

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/dufourc1/pygraphon.git
$ cd pygraphon
$ pip install -e .
```
</details>

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/dufourc1/pygraphon/blob/master/CONTRIBUTING.rst) for more information on getting involved.

## 👋 Attribution

<details>
 <summary>See Attributions</summary>


### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.
</details>

## 🛠️ For Developers

<details>
	<summary>See developer instructions</summary>


The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `uv` (see [uv installation](https://docs.astral.sh/uv/)), the unit tests in the `tests/` folder can be run reproducibly with:

```shell
$ uv run pytest tests
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/dufourc1/pygraphon/actions?query=workflow%3ATests).

### 📝 Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/en/master/). After installing the package in development mode, the documentation can be built locally with:

```shell
$ cd docs
$ make html
```

The documentation will then be available in `docs/build/html/`.

To build the documentation, you'll need to install the necessary dependencies. This can be done with:

```shell
$ uv sync --all-extras
```

note: to correctly format the documentation, one can use tool such as `rstfmt` (installable with `uv pip install rstfmt`).
</details>
