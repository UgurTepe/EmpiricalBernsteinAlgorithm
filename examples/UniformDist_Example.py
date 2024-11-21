import sys
import numpy as np

# Importing own functions
sys.path.append("./")

from src.algorithms import EBS
from src.algorithms import hoeffding_bound as tmin#

'''
Simple example of EBS using a batched Uniform(a,b,l) distribution 
'''

a = 0
b = 1
delta = 0.1
epsilon = 0.01
l = 10 #Varying this adjust the variance, while keeping the mean the same; sigma² = (b-a)²/(12l)

alg = EBS(epsilon=epsilon, delta=delta, range_of_rndvar=b - a)

# Stops when condition is met
while alg.cond_check():
    sample = np.random.uniform(a, b, (1, l))
    sample = np.mean(sample, axis=1)[0]
    alg.add_sample(sample)


print("========================================")
print(" Empirical Bernstein Stopping Algorithm ")
print("========================================")
print(f"# Samples: EBS:       {alg.get_numsamples()}")
print(f"           Höffding:  {tmin(epsilon=epsilon, delta=delta, rng=b-a)}")
print("----------------------------------------")
print(f"Estimate Mean:        {alg.get_estimate()}")
print(f"Actual Mean:          {(a+b)/2}")
print("----------------------------------------")
print(f"|Estimate - mu| <= Epsilon ? --> {np.abs((alg.get_estimate()-(a+b)/2)) <= epsilon}")
print("========================================")

