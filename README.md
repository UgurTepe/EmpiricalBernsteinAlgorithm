# Empirical Bernstein Stopping (EBS) Algorithm

EBS

- [Empirical Bernstein Stopping (EBS) Algorithm](#empirical-bernstein-stopping-ebs-algorithm)
- [Overview](#overview)
- [Documentation](#documentation)
- [Python Dependencies](#python-dependencies)
- [Example](#example)
- [Data for Examples](#data-for-examples)
- [License](#license)


# Overview
This repository contains the Empirical Bernstein stopping algorithm in its original form, see [paper](https://www.cs.toronto.edu/~vmnih/docs/ebstop.pdf), our **modified version** and code for calculating the Höffding's bound.

# Documentation
See [notebook](/tutorial.ipynb) or directly inside the code.

# Python Dependencies
EBS depends on a several Python libraries.
One may recreate the conda environment using the `.yml` [file](/req_conda.yml).
The environment can then be created with the command:
```
conda env create -f req_conda.yml
```
Activate the environment `EBS`:
```
conda activate EBS
```
Alternatively see the [requirements.txt](/requirements.txt) for the required packages.
# Example
Here is a quick example of how to use the algorithm to estimate a random variable.
For example, the random variable could be sampled from a uniform distribution.

```python
# Importing packages
import numpy as np
from algorithms import EBS
a = 0
b = 1
delta = 0.1
epsilon = 0.01

ebs = EBS(epsilon=epsilon,delta=delta,range_of_rndvar=b - a)

while ebs.cond_check():
    ebs.add_sample(np.random.uniform(a,b))

print('Estimate: ',ebs.get_estimate())
print('No. Samples:',round(ebs.get_numsamples()))
>>> Estimate: 0.50
>>> No. Samples: 14978
```
# Data for Examples
For one of the example files, additional data is required.
As the files are fairly large ~400mb, they are not included in this repository but can be downloaded on [figshare]((https://doi.org/10.6084/m9.figshare.27879525.v)) and inserted into the following path:
```dep/samples_gs/```

# License
This project is covered under the **Apache 2.0 License**.

[def]: #license
