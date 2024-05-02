import sys
import numpy as np
import matplotlib.pyplot as plt

'''
Configuring matplotlib
'''
plt.rcParams["figure.autolayout"] = True
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.size': 15})

# Importing own functions
sys.path.append("./")
from dep.qm_vqe import *


g = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910] # d = 0.75 Å
arr_par, arr_energy, arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf, arr_max_flag = vqe_adam_h2(eps_bern=0.1,delta=0.1,hamiltonian_coeff=g)

print(np.sum(arr_steps))
plt.plot(arr_par, arr_est_energy, '.-', label='Energy')
plt.xlabel(r'$\theta$')
plt.ylabel(r'Energy (Hartee)')
ground_energy = np.linalg.eigvalsh(h2_op(g))[0]
plt.axhline(y=ground_energy, color='r', linestyle='--',label = r'Ground Energy for $d = 0.75 \AA$')
plt.fill_between(arr_par, ground_energy-0.1, ground_energy+0.1,facecolor = 'red',alpha = 0.4,label = r'$\pm \epsilon$')
plt.grid()
plt.legend()
plt.show()

