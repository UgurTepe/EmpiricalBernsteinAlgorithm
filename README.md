# Empirical Bernstein Stopping (EBS) Algorithm

EBS

- [Overview](#overview)
- [Documentation](#documentation)
- [Python Dependencies](#Python-Dependencies)
- [Example](#Example)
- [License](#license)


# Overview
``EBS`` 
Collecting of algorithms based on this [[paper](https://www.cs.toronto.edu/~vmnih/docs/ebstop.pdf)], adapted to absolute errors and implemented in a Python class structure.

# Documentation
See [notebook](/tutorial.ipynb) or inside code.

# Python Dependencies
`EBS` depends on a plethora of Python scientific libraries which can be found in the [requirements.txt](/requirements.txt) file.
For the exmamples more packages may be needed, see [requirements_examples.txt](/requirements_examples.txt) file.

# Example
Here is a quick example of how to use a EBS algorithm.

```python
# Importing packages
import numpy, algorithms
a = 0
b = 1
delta = 0.1
epsilon = 0.01

ebs = eba_geo_marg(epsilon=epsilon,delta=delta,range_of_rndvar=b - a)

while ebs.cond_check():
    ebs.add_sample(np.random.uniform(a,b))
    ebs.inner_cond_check()

print('Estimate: ',ebs.get_estimate())
print('No. Samples:',round(ebs.get_step()))
>>> Estimate: 0.5002286837992509
>>> No. Samples: 2254
```
# License
This project is covered under the **Apache 2.0 License**.

