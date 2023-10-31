
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configuring matplotlib
from matplotlib import rc
plt.rcParams["figure.autolayout"] = True
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.size': 15})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Importing own functions
sys.path.append("./")
from dep.qm_vqe import *



'''
VQE with variable epsilon
'''
arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag = vqe_eps(0.1)

# Default g --> d = 0.75
g = [0.2252, 0.3435, -0.4347,0.5716,0.0910, 0.0910]

plt.plot(arr_par1, arr_energy, label='Energy')
plt.xlabel(r'$\theta$')
plt.ylabel(r'Energy (Hartee)')
plt.axhline(y=np.linalg.eigvalsh(h2_op(g))[0], color='r',linestyle = '--')
plt.grid()
plt.legend()
plt.show()

'''
VQE with variable g
'''
# paras = np.loadtxt(fname='config/h2_parameter.csv',dtype=None,delimiter=',',skiprows=1)

# arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag = vqe_g(paras[1])

