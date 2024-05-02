

import numpy as np
from numpy import linalg as LA
import sys

sys.path.append("./")
from dep.qm_vqe import *
from dep.qm_gates import *
from dep.qm_tools import *
from dep.tools import AdamOptimizer
from dep.qm_circuit import *
from dep.shadow_grouping import *
from src.algorithms import eba_geo_marg as bernstein
from src.algorithms import hoeffding_bound

def vqe_non_adam_h2(hamiltonian_coeff,delta = 0.1,eps_bern= 0.1,n = 700,alpha = 0.1,eps = 0.1):
    par1 = -0.5  # Initial parameters
    range_g = 2*np.sum(np.abs(hamiltonian_coeff[1:]))
    max_sample = hoeffding_bound(delta= delta,rng=range_g,epsilon=eps_bern)
    E_corr = np.linalg.eigvalsh(h2_op(hamiltonian_coeff))[0]

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
        state_normal = circ(par1)

        # SPSA method
        rnd1 = np.random.choice([-1, 1])
        state_shift_plus = circ(
            par1 + eps*rnd1)
        state_shift_minus = circ(
        par1 - eps*rnd1)

        '''
        Loops for EBS algorithm
        '''
        # Resets EBS every outer loop iteration
        ebs = bernstein(delta=delta, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_plus = bernstein(delta=delta, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_minus = bernstein(delta=delta, epsilon=eps_bern, range_of_rndvar=range_g)

        '''
        Prepare Samples for energy and gradient estimatation
        '''
        
        # EBS
        while ebs.cond_check() and 3*ebs.get_step() < max_sample:
            ebs.add_sample(h2_measure(state_normal,hamiltonian_coeff))
            ebs.inner_cond_check()
            
        # Saving values
        it_normal = 3*ebs.get_step()
        #it_l1 = ebs_l1.get_step()
        est_energy = ebs.get_estimate()
        est_variance = ebs.get_var()[-1]
        energy = expected_value(state_normal, h2_op(hamiltonian_coeff))    
        variance = expected_value(state_normal, np.linalg.matrix_power(
            h2_op(hamiltonian_coeff), 2)) - expected_value(state_normal, h2_op(hamiltonian_coeff))**2

        # EBS for Energy(x+εΔ,y+εΔ)
        while ebs_shift_plus.cond_check() and 3*ebs_shift_plus.get_step() < max_sample:
            ebs_shift_plus.add_sample(h2_measure(state_shift_plus,hamiltonian_coeff))
            ebs_shift_plus.inner_cond_check()
        energy_shifted_plus = ebs_shift_plus.get_estimate()
        
        # EBS for Energy(x-εΔ,y-εΔ)
        while ebs_shift_minus.cond_check() and 3*ebs_shift_minus.get_step() < max_sample:
            ebs_shift_minus.add_sample(h2_measure(state_shift_minus,hamiltonian_coeff))
            ebs_shift_minus.inner_cond_check()
        energy_shifted_minus = ebs_shift_minus.get_estimate()

        # Save data
        arr_energy.append(energy)
        arr_est_energy.append(est_energy)
        arr_var.append(variance)
        arr_est_var.append(est_variance)
        arr_par1.append(par1)
        arr_steps.append(it_normal)
        arr_höf.append(max_sample)
        arr_max_flag.append(flag*1)
        
        # Estimate the Gradient via SPSA method
        grad1 = (energy_shifted_plus-energy_shifted_minus) / (2*eps*rnd1)

        # Updating the Parameters via Gradient Descent method
        par1 -= alpha*grad1
        
        if flag:
            break
        if np.abs(energy - E_corr) <= eps_bern:
            flag = True
    return arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag

def vqe_adam_h2(hamiltonian_coeff,delta = 0.1,eps_bern= 0.1,max_it = 500,alpha = 0.1,eps = 0.1):
    """
    Performs the Variational Quantum Eigensolver (VQE) algorithm using the Adam optimizer for a given Hamiltonian.

    Parameters:
    - hamiltonian_coeff (list): Coefficients of for given bound length d
    - delta (float): Confidence parameter for the Empirical Bernstein Sampling (EBS) algorithm. Default is 0.1.
    - eps_bern (float): Error parameter for the EBS algorithm. Default is 0.1.
    - max_it (int): Number of iterations for the VQE algorithm. Default is 500.
    - alpha (float): Learning rate for the (Adam) optimizer. Default is 0.1.
    - eps (float): Stepsize for Gradient Estimation. Default is 0.1.

    Returns:
    - arr_par1 (list): List of parameter values at each iteration.
    - arr_energy (list): List of energy values at each iteration.
    - arr_var (list): List of variance values at each iteration.
    - arr_est_energy (list): List of estimated energy values at each iteration.
    - arr_est_var (list): List of estimated variance values at each iteration.
    - arr_steps (list): List of step counts at each iteration.
    - arr_höf (list): List of maximum sample counts at each iteration.
    - arr_max_flag (list): List of flag values indicating convergence at each iteration.

    """
    par1 = -1  # Initial parameters
    range_g = 2*np.sum(np.abs(hamiltonian_coeff[1:]))
    max_sample = hoeffding_bound(delta= delta,rng=range_g,epsilon=eps_bern)
    E_corr = np.linalg.eigvalsh(h2_op(hamiltonian_coeff))[0]

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
    # initialize adam optimizers
    adam = AdamOptimizer(par1, alpha=alpha)

    for i in range(max_it):
        # set states to |01>
        # Apply circuit on state
        state_normal = circ(par1)

        # SPSA method
        rnd1 = np.random.choice([-1, 1])
        state_shift_plus = circ(
            par1 + eps*rnd1)
        state_shift_minus = circ(
        par1 - eps*rnd1)

        '''
        Loops for EBS algorithm
        '''
        # Resets EBS every outer loop iteration
        ebs = bernstein(delta=delta, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_plus = bernstein(delta=delta, epsilon=eps_bern, range_of_rndvar=range_g)
        ebs_shift_minus = bernstein(delta=delta, epsilon=eps_bern, range_of_rndvar=range_g)

        '''
        Prepare Samples for energy and gradient estimatation
        '''
        
        # EBS
        while ebs.cond_check() and 3*ebs.get_step() < max_sample:
            ebs.add_sample(h2_measure(state_normal,hamiltonian_coeff))
            ebs.inner_cond_check()
            
        # Saving values
        it_normal = 3*ebs.get_step()
        #it_l1 = ebs_l1.get_step()
        est_energy = ebs.get_estimate()
        est_variance = ebs.get_var()[-1]
        energy = expected_value(state_normal, h2_op(hamiltonian_coeff))    
        variance = expected_value(state_normal, np.linalg.matrix_power(
            h2_op(hamiltonian_coeff), 2)) - expected_value(state_normal, h2_op(hamiltonian_coeff))**2

        # EBS for Energy(x+εΔ,y+εΔ)
        while ebs_shift_plus.cond_check() and 3*ebs_shift_plus.get_step() < max_sample:
            ebs_shift_plus.add_sample(h2_measure(state_shift_plus,hamiltonian_coeff))
            ebs_shift_plus.inner_cond_check()
        energy_shifted_plus = ebs_shift_plus.get_estimate()
        
        # EBS for Energy(x-εΔ,y-εΔ)
        while ebs_shift_minus.cond_check() and 3*ebs_shift_minus.get_step() < max_sample:
            ebs_shift_minus.add_sample(h2_measure(state_shift_minus,hamiltonian_coeff))
            ebs_shift_minus.inner_cond_check()
        energy_shifted_minus = ebs_shift_minus.get_estimate()

        # Save data
        arr_energy.append(energy)
        arr_est_energy.append(est_energy)
        arr_var.append(variance)
        arr_est_var.append(est_variance)
        arr_par1.append(par1)
        arr_steps.append(it_normal)
        arr_höf.append(max_sample)
        arr_max_flag.append(flag*1)
        
        # Estimate the Gradient via SPSA method
        grad1 = (energy_shifted_plus-energy_shifted_minus) / (2*eps*rnd1)
        
        # Adam optimizer
        par1 = adam.updated_alpha(grad1)

        if flag:
            break
        if np.abs(energy - E_corr) <= eps_bern:
            flag = True
    return arr_par1, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag

