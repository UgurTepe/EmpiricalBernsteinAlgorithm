import sys
import numpy as np

# Importing own functions
sys.path.append("./")

from dep.tools import num_after_point, trunc
from src.algorithms import *



'''
Simple example of EBS using a Uniform(a,b) distribution
'''
a = 0
b = 1
delta = 0.1
epsilon = 0.1
l = 10 #
alg = eba_mod(epsilon=epsilon, delta=delta, range_of_rndvar=b - a)

# Stops when condition is met
while alg.cond_check():
    sample = np.random.uniform(a, b, (1, l))
    sample = np.mean(sample, axis=1)[0]
    alg.add_sample(sample)


print('# Samples: EBS:      ', round(alg.get_step()))
print('           Höffding: ', round(
    hoeffding_bound(epsilon=epsilon, delta=delta, rng=b-a)))
print('Estimate Mean: ', trunc(alg.get_estimate(),num_after_point(epsilon)))
print('Actual Mean: ', trunc((a+b)/2, num_after_point(epsilon)))
print('Estimated Variance: ', trunc(alg.get_var()[-1],num_after_point(epsilon)))
print('Actual Variance: ', trunc((b-a)**2/(12*l),num_after_point(epsilon)))
print('|Estimate - mu| <= Epsilon ? -->', np.abs((alg.get_estimate()-(a+b)/2)) <= epsilon)

