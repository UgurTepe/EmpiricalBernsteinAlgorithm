import sys
import numpy as np

# Importing own functions
sys.path.append("./")
from src.algorithms import *
from dep.tools import num_after_point, trunc

'''
Simple example of EBS using a Uniform(a,b) distribution
'''
a = 0
b = 1
delta = 0.1
epsilon = 0.01

alg = eba_geo_marg(epsilon=epsilon,delta=delta,range_of_rndvar=b - a)

#Stops when condition is met
while alg.cond_check():
    alg.add_sample(np.random.uniform(a,b))
    alg.inner_cond_check()


print('# Samples: EBS:      ',round(alg.get_step()))
print('           Höffding: ',round(hoeffding_bound(epsilon=epsilon,delta=delta,rng=b-a)))
print('Estimate: ',alg.get_estimate())
print('Expected Value: ',trunc((a+b)/2,num_after_point(epsilon)))
print('|Estimate - mu| <= Epsilon ?',np.abs((alg.get_estimate()-(a+b)/2))<= epsilon)
# print('Variance: ',(b-a)**2/12)
