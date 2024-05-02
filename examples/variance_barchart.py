import sys
import numpy as np
import matplotlib.pyplot as plt

# Importing own functions
sys.path.append("./")
import src.algorithms as alg

'''
Parameter for the distribution & stopping algorithms
'''
a = 0
b = 1
eps = 0.01
delta = 0.1
rng = b - a

l_arr = [1, 5, 10, 50, 100, 1000]

eba_arr = []
eba_geo_arr = []
eba_marg_arr = []

for i, l in enumerate(l_arr):
    eba = alg.eba_simple(epsilon=eps, delta=delta, range_of_rndvar=rng)
    eba_geo = alg.eba_geo(epsilon=eps, delta=delta, range_of_rndvar=rng)
    eba_marg = alg.eba_geo_marg(epsilon=eps, delta=delta, range_of_rndvar=rng)

    # Stops when condition is met
    while eba.cond_check():
        # Get random variable that is uniformly dist.
        sample = np.random.uniform(a, b, (1, l))
        sample = np.mean(sample, axis=1)[0]
        eba.add_sample(sample)
    eba_arr.append(eba.get_step())

    while eba_geo.cond_check():
        # Get random variable that is uniformly dist.
        sample = np.random.uniform(a, b, (1, l))
        sample = np.mean(sample, axis=1)[0]
        eba_geo.add_sample(sample)
        if eba_geo.inner_cond_check():
            eba_geo.update_ct()
    eba_geo_arr.append(eba_geo.get_step())

    while eba_marg.cond_check():
        # Get random variable that is uniformly dist.
        sample = np.random.uniform(a, b, (1, l))
        sample = np.mean(sample, axis=1)[0]
        eba_marg.add_sample(sample)
        if eba_marg.inner_cond_check():
            eba_marg.update_ct()
    eba_marg_arr.append(eba_marg.get_step())
    hof_arr = np.ones_like(
        eba_arr)*alg.hoeffding_bound(delta=delta, epsilon=eps, rng=rng)


dist = 0.25
wid = 0.1
br1 = np.arange(len(hof_arr))
br2 = [x + dist for x in br1]
br3 = [x + dist for x in br2]
br4 = [x + dist for x in br3]

plt.bar(br1, hof_arr, color='k', width=wid,
        edgecolor='grey', label="Hoeffding's Bound")

plt.bar(br2, eba_arr, color='r', width=wid,
        edgecolor='grey', label='EBA')

plt.bar(br3, eba_geo_arr, color='g', width=wid,
        edgecolor='grey', label='EBA Geo')

plt.bar(br4, eba_marg_arr, color='b', width=wid,
        edgecolor='grey', label='EBA Geo + Anytime Stop')


plt.ylabel('Number of Samples')
plt.yscale('log')
plt.xticks([r + dist for r in range(len(hof_arr))],
           l_arr)
plt.title(rf'U(0,1,l) | $\epsilon = {eps}$ | $\delta = {delta}$')
plt.legend()
plt.show()
