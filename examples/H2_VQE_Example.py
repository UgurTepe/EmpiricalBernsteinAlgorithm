import sys
import numpy as np
import matplotlib.pyplot as plt

# Importing own functions
sys.path.append("./")
from dep.qm_vqe import *

# Set the style
plt.style.use('seaborn-v0_8-deep')
eps = 0.01
delta = 0.1

g = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910] # d = 0.75 Å
arr_par1, arr_energy,arr_var, est_energy, est_variance, arr_steps, arr_höf, arr_höf_adap,arr_grad1,arr_momentum,arr_epsilon = VQE_H2(eps_bern=eps,delta=delta,hamiltonian_coeff=g,adaptive_eps=False)

arr_par1, arr_energy, arr_var, arr_steps, arr_höf, arr_höf_adap, arr_grad1, arr_momentum, arr_epsilon = \
    [arr[:-1] for arr in [arr_par1, arr_energy, arr_var, arr_steps, arr_höf, arr_höf_adap, arr_grad1, arr_momentum, arr_epsilon]]

arr_energy[-1] = est_energy
plt.plot(arr_par1, arr_energy, 'x-', label='Energy')
plt.plot(arr_par1[-1], arr_energy[-1], 'x',color = 'green', label='Estimated Ground State Energy')
plt.xlabel(r'$\theta$')
plt.ylabel(r'Energy (Hartee)')
ground_energy = np.linalg.eigvalsh(h2_op(g))[0]
plt.axhline(y=ground_energy, color='r', linestyle='--',label = r'Ground Energy for $d = 0.75 \AA$')
plt.fill_between(arr_par1, ground_energy-eps, ground_energy+eps,facecolor = 'red',alpha = 0.4,label = r'$\pm \epsilon$')
plt.grid()
plt.legend()
plt.show()

