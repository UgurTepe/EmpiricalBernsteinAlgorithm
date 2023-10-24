
import sys
sys.path.append("./")
from dep.qm_vqe import *

#paras = np.loadtxt(fname='config/h2_parameter.csv',dtype=None,delimiter=',',skiprows=1)
eps = [0.2]
for n,param in enumerate(eps):
    print(f'{n} --- epsilon={param}')
    print(vqe_eps(param))