
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

paras = np.loadtxt(fname='config/h2_parameter.csv',dtype=None,delimiter=',',skiprows=1)


arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag = vqe_eps(0.2)

plt.plot(arr_par1, arr_energy, label='Energy')
plt.xlabel(r'$\theta$')
plt.ylabel(r'Energy (Hartee)')
plt.grid()
plt.legend()
plt.show()