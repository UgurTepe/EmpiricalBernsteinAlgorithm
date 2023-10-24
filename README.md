# Empirical Bernstein Stopping Algorithm

`shadowgrouping` is a Python package containing the measurement scheme `ShadowGrouping` of Ref. https://arxiv.org/abs/2301.03385. In addition, the used numerical benchmarks of previous state-of-the-art methods have been unified within this package.

- [Overview](#overview)
- [Documentation](#documentation)
- [Python Dependencies](#Python-Dependencies)
- [Installation Guide](#installation-guide)
- [Using the environment](#using-the-environment)
- [License](#license)

# Overview
``shadowgrouping`` 
The package utilizes a simple class structure for the measurement schemes as well the various energy estimators that come along with them.
The package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows) from this GitHub repo, see below.

# Documentation
There is no official documentation, but all classes within the package have been documented individually.
We refer to the `tutorial.ipynb` for usage of the package.

# Python Dependencies
`shadowgrouping` depends on a plethora of Python scientific libraries which can be found in `requirements.txt`.

# Using the package:
- To run demo notebooks:
  - `jupyter notebook`
  - Then copy the url it generates, it looks something like this: `http://localhost:8889/?token=dde30ccc772afed3012e7c3be67a537cc1ea9036c22357c8`
  - Open it in your browser
  - Then open `tutorial.ipynb` which includes the minimal working example. Running all executable code in the notebooks sequentially should not take more than a few minutes on a standard laptop. If other molecules are selected, however, this run time can easily turn into a few hours though.

# License

This project is covered under the **Apache 2.0 License**.
