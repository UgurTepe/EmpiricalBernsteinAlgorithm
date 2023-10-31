

import numpy as np
from numpy import linalg as LA
import sys

sys.path.append("./")
from dep.qm_vqe import *
from dep.qm_gates import *
from dep.qm_tools import *
from dep.qm_circuit import *
from dep.shadow_grouping import *
from src.algorithms import eba_geo_marg as bernstein
from src.algorithms import hoeffding_bound

def vqe_g(hamiltonian_coeff):
    par1 = -0.1  # Initial parameters
    alpha = 0.1  # Gradient descent
    eps = 0.1
    eps_bern = 0.01
    max_sample = 10**8  # Max number of samples for EBS
    range_g = 2*np.sum(np.abs(hamiltonian_coeff[2:]))
    n = 500
    '''
    Initialization of Arrays and Variables 
    '''
    energy = 0
    est_energy = 0
    variance = 0
    energy_shifted_plus = 0
    energy_shifted_minus = 0
    grad1 = 0
    arr_energy = []
    arr_est_energy = []
    arr_var = []
    arr_est_var = []
    arr_par1 = []
    arr_steps = []
    arr_höf = []
    flag = False
    arr_max_flag = []
    
    for i in range(n):
          
        # set states to |01>
        # Apply circuit on state
        state_normal = h2(par1, np.array([0, 1, 0, 0]))

        # SPSA method
        rnd1 = np.random.choice([-1, 1])
        state_shift_plus = h2(
            par1 + eps*rnd1, np.array([0, 1, 0, 0]))
        state_shift_minus = h2(
        par1 - eps*rnd1, np.array([0, 1, 0, 0]))

        '''
        Loops for EBS algorithm
        '''
        # Resets EBS every outer loop iteration
        ebs = bernstein(delta=0.1, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_plus = bernstein(delta=0.1, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_minus = bernstein(delta=0.1, epsilon=eps_bern, range_of_rndvar=range_g)

        '''
        Prepare Samples for energy and gradient estimatation
        '''
        
        # EBS
        while ebs.cond_check():
            ebs.add_sample(h2_measure(state_normal,hamiltonian_coeff[1:]))
            if ebs.inner_cond_check():
                ebs.update_ct()
            if ebs.get_step() > max_sample:
                break  
            
        # Saving values
        it_normal = 3*ebs.get_step()
        #it_l1 = ebs_l1.get_step()
        est_energy = ebs.get_estimate()
        est_variance = ebs.get_var()[-1]
        energy = expected_value(state_normal, h2_op(hamiltonian_coeff[1:]))    
        variance = expected_value(state_normal, np.linalg.matrix_power(
            h2_op(hamiltonian_coeff[1:]), 2)) - expected_value(state_normal, h2_op(hamiltonian_coeff[1:]))**2

        # EBS for Energy(x+εΔ,y+εΔ)
        while ebs_shift_plus.cond_check():
            ebs_shift_plus.add_sample(h2_measure(state_shift_plus,hamiltonian_coeff[1:]))
            if ebs_shift_plus.inner_cond_check():
                ebs_shift_plus.update_ct()
            if ebs_shift_plus.get_step() > max_sample:
                break  
        energy_shifted_plus = ebs_shift_plus.get_estimate()
        
        # EBS for Energy(x-εΔ,y-εΔ)
        while ebs_shift_minus.cond_check():
            ebs_shift_minus.add_sample(h2_measure(state_shift_minus,hamiltonian_coeff[1:]))
            if ebs_shift_minus.inner_cond_check():
                ebs_shift_minus.update_ct()
            if ebs_shift_minus.get_step() > max_sample:
                break
        energy_shifted_minus = ebs_shift_minus.get_estimate()


        # Save data
        arr_energy.append(energy)
        arr_est_energy.append(est_energy)
        arr_var.append(variance)
        arr_est_var.append(est_variance)
        arr_par1.append(par1)
        arr_steps.append(it_normal)
        arr_höf.append(hoeffding_bound(0.1,range_g,eps_bern))
        arr_max_flag.append(flag*1)
        
        # Estimate the Gradient via SPSA method
        grad1 = (energy_shifted_plus-energy_shifted_minus) / (2*eps*rnd1)

        # Updating the Parameters via Gradient Descent method
        par1 -= alpha*grad1
        
        if flag:
            break
        if np.abs(energy - np.linalg.eigvalsh(h2_op(hamiltonian_coeff[1:]))[0]) <= 0.01:
            flag = True
    return arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag

def vqe_eps(eps_0):
    g = [0.2252, 0.3435, -0.4347,0.5716,0.0910, 0.0910]
    par1 = -0.1  # Initial parameters
    alpha = 0.1  # Gradient descent
    eps = 0.1
    eps_bern = eps_0
    range_g = 2*np.sum(np.abs(g[1:]))
    max_sample = 10**8  # Max number of samples for EBS
    n = 500
    '''
    Initialization of Arrays and Variables 
    '''
    energy = 0
    est_energy = 0
    variance = 0
    energy_shifted_plus = 0
    energy_shifted_minus = 0
    grad1 = 0
    arr_energy = []
    arr_est_energy = []
    arr_var = []
    arr_est_var = []
    arr_par1 = []
    arr_steps = []
    arr_l1 = []
    arr_höf = []
    flag = False
    arr_max_flag = []
    
    for i in range(n):
          
        # set states to |01>
        # Apply circuit on state
        state_normal = h2(par1, np.array([0, 1, 0, 0]))

        # SPSA method
        rnd1 = np.random.choice([-1, 1])
        state_shift_plus = h2(
            par1 + eps*rnd1, np.array([0, 1, 0, 0]))
        state_shift_minus = h2(
        par1 - eps*rnd1, np.array([0, 1, 0, 0]))

        '''
        Loops for EBS algorithm
        '''
        # Resets EBS every outer loop iteration
        ebs = bernstein(delta=0.1, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_plus = bernstein(delta=0.1, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_minus = bernstein(delta=0.1, epsilon=eps_bern, range_of_rndvar=range_g)

        '''
        Prepare Samples for energy and gradient estimatation
        '''

        # Naive
        while ebs.cond_check():
            ebs.add_sample(h2_measure(state_normal,g))
            if ebs.inner_cond_check():
                ebs.update_ct()
            if ebs.get_step() > max_sample:
                break  
            
        # Saving values
        it_normal = 3*ebs.get_step()
        #it_l1 = ebs_l1.get_step()
        est_energy = ebs.get_estimate()
        est_variance = ebs.get_var()[-1]
        energy = expected_value(state_normal, h2_op(g))    
        variance = expected_value(state_normal, np.linalg.matrix_power(
            h2_op(g), 2)) - expected_value(state_normal, h2_op(g))**2

        # EBS for Energy(x+εΔ,y+εΔ)
        while ebs_shift_plus.cond_check():
            ebs_shift_plus.add_sample(h2_measure(state_shift_plus,g))
            if ebs_shift_plus.inner_cond_check():
                ebs_shift_plus.update_ct()
            if ebs_shift_plus.get_step() > max_sample:
                break  
        energy_shifted_plus = ebs_shift_plus.get_estimate()
        
        # EBS for Energy(x-εΔ,y-εΔ)
        while ebs_shift_minus.cond_check():
            ebs_shift_minus.add_sample(h2_measure(state_shift_minus,g))
            if ebs_shift_minus.inner_cond_check():
                ebs_shift_minus.update_ct()
            if ebs_shift_minus.get_step() > max_sample:
                break
        energy_shifted_minus = ebs_shift_minus.get_estimate()


        # Save data
        arr_energy.append(energy)
        arr_est_energy.append(est_energy)
        arr_var.append(variance)
        arr_est_var.append(est_variance)
        arr_par1.append(par1)
        arr_steps.append(it_normal)
        arr_höf.append(hoeffding_bound(0.1,range_g,eps_bern))
        arr_max_flag.append(flag*1)
        
        # Estimate the Gradient via SPSA method
        grad1 = (energy_shifted_plus-energy_shifted_minus) / (2*eps*rnd1)

        # Updating the Parameters via Gradient Descent method
        par1 -= alpha*grad1
        
        if flag:
            break
        if np.abs(energy - np.linalg.eigvalsh(h2_op(g))[0]) <= eps_bern:
            flag = True
            
    return arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag
