<!--
<p align="center">
	<img src="https://github.com//pygraphon/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
	PyGraphon
</h1>

<p align="center">
		<a href="https://github.com/dufourc1/pygraphon/actions/workflows/build.yml">
			<img alt="Build" src="https://github.com/dufourc1/pygraphon/workflows/build/badge.svg" />
		</a>
		<a href="https://github.com/cthoyt/cookiecutter-python-package">
				<img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-python--package-yellow" /> 
		</a>
		<a href="https://pypi.org/project/pygraphon">
				<img alt="PyPI" src="https://img.shields.io/pypi/v/pygraphon" />
		</a>
		<a href="https://pypi.org/project/pygraphon">
				<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/pygraphon" />
		</a>
		<a href="https://github.com/dufourc1/pygraphon/blob/master/LICENSE">
				<img alt="PyPI - License" src="https://img.shields.io/pypi/l" />
		</a>
		<a href='https://pygraphon.readthedocs.io/en/latest/?badge=latest'>
				<img src='https://readthedocs.org/projects/pygraphon/badge/?version=latest' alt='Documentation Status' />
		</a>
		<a href='https://github.com/psf/black'>
				<img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
		</a>
</p>

Welcome to the documentation of our Python library for estimating graphons from observed data! Our library provides a powerful set of tools for estimating graphons, which are a type of object used to model large, random graphs, from observed data.

Graphons are a powerful tool for modeling real-world networks, as they can capture the complex, often random structure of these networks in a flexible and scalable way. However, estimating graphons from observed data can be a challenging task, especially when dealing with large and complex networks.

Our library specializes in providing algorithms and tools for estimating graphons from observed data, making it easier to work with these objects in a practical setting. We provide a range of estimation methods that are well-suited to different types of data, including methods based on moments, spectral methods, and maximum likelihood estimation.

Our library also provides tools for preprocessing data, such as data normalization, denoising, and data discretization. These preprocessing tools can help improve the accuracy and reliability of graphon estimation methods, especially when dealing with noisy or incomplete data.

In addition to estimation, our library also provides tools for working with estimated graphons, including visualization tools and algorithms for computing various graph statistics from estimated graphons.

By providing a powerful set of tools for estimating graphons from observed data, our library can help researchers and practitioners gain new insights into the structure and behavior of complex networks. Whether you're analyzing social networks, biological networks, or transportation networks, our library has something to offer.

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
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

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com//pygraphon/blob/master/CONTRIBUTING.rst) for more information on getting involved.

## 👋 Attribution

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

## 🛠️ For Developers

<details>
	<summary>See developer instrutions</summary>

	
The final section of the README is for if you want to get involved by making a code contribution.

### ❓ Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox -q 
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com//pygraphon/actions?query=workflow%3ATests).

### 📝 Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/en/master/). After installing the package in development mode, the documentation can be built locally with:

```shell
$ tox -e docs
```

The documentation will then be available in `.tox/tmp/build/html/`.


Another way to build the documentation is to use the `make` command:

```shell
$ cd docs
$ make html
```

The documentation will then be available in `docs/build/html/`. To use the `make` command, one needs to install the
*additional* necessary dependencies in the virutal environment. This can be done with:

```shell
$ pip install -r docs/requirements.txt
```

note: to correctly format the documentation, one can use tool such as `rstfmt` (installable with `pip install rstfmt`).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses BumpVersion to switch the version number in the `setup.cfg` and
	 `src/pygraphon/version.py` to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel
3. Uploads to PyPI using `twine`. Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
	 step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
	 use `tox -e bumpversion minor` after.
</details>
