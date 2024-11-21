import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Importing own functions
sys.path.append("./")

from src.algorithms import hoeffding_bound as tmin
from src.algorithms import EBS_UnModded as EBS_UnMod
from src.algorithms import EBS

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
eba_unmod_arr = []

for i, l in enumerate(l_arr):
        eba = EBS(epsilon=eps, delta=delta, range_of_rndvar=rng)
        eba_unmod = EBS_UnMod(epsilon=eps, delta=delta, range_of_rndvar=rng)

        # Stops when condition is met
        while eba.cond_check():
                # Get random variable that is uniformly dist.
                sample = np.random.uniform(a, b, (1, l))
                sample = np.mean(sample, axis=1)[0]
                eba.add_sample(sample)
        eba_arr.append(eba.get_numsamples())

        while eba_unmod.cond_check():
                # Get random variable that is uniformly dist.
                sample = np.random.uniform(a, b, (1, l))
                sample = np.mean(sample, axis=1)[0]
                eba_unmod.add_sample(sample)
                
        eba_unmod_arr.append(eba_unmod.get_numsamples())
        hof_arr = np.ones_like(
                eba_arr)*tmin(delta=delta, epsilon=eps, rng=rng)

# Set the style
plt.style.use('seaborn-v0_8-deep')

dist = 0.25
wid = 0.1
br1 = np.arange(len(hof_arr))
br2 = [x + dist for x in br1]
br3 = [x + dist for x in br2]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(br1, hof_arr, color='navy', width=wid,
           edgecolor='grey', label="Hoeffding's Bound")

ax.bar(br2, eba_arr, color='darkorange', width=wid,
           edgecolor='grey', label='EBA')

ax.bar(br3, eba_unmod_arr, color='forestgreen', width=wid,
           edgecolor='grey', label='EBA UnModded')

ax.set_ylabel('Number of Samples')
ax.set_yscale('log')
ax.set_xticks([r + dist for r in range(len(hof_arr))])
ax.set_xticklabels(l_arr)
ax.set_title(rf'U(0,1,l) | $\epsilon = {eps}$ | $\delta = {delta}$')
ax.set_xlabel('Batch Size l')
ax.legend()

# Add grid lines
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Customize the ticks
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.tick_params(axis='y', which='both', length=0)

plt.show()
