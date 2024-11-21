

import numpy as np
from numpy import linalg as LA
import sys

sys.path.append("./")
from dep.qm_vqe import *
from dep.qm_gates import *
from dep.qm_tools import *
from dep.tools import AdamOptimizer
from dep.qm_circuit import *
from src.algorithms import EBS
from src.algorithms import hoeffding_bound

def VQE_H2(hamiltonian_coeff,delta = 0.1,eps_bern= 0.1,max_it = 500,alpha = 0.1,eps = 0.1,gamma1=1.25,gamma2=0.75,adaptive_eps = False):
    """
        Variational Quantum Eigensolver (VQE) for H2 molecule using the Empirical Bernstein Stopping (EBS) algorithm.
        Parameters:
        -----------
        hamiltonian_coeff : array-like
            Coefficients of the Hamiltonian for the H2 molecule.
        delta : float, optional
            Confidence level for the Hoeffding bound and EBS algorithm (default is 0.1).
        eps_bern : float, optional
            Desired accuracy for the final energy estimation (default is 0.1).
        max_it : int, optional
            Maximum number of iterations for the optimization loop (default is 500).
        alpha : float, optional
            Learning rate for the Adam optimizer (default is 0.1).
        eps : float, optional
            Small shift for finite difference gradient estimation (default is 0.1).
        gamma1 : float, optional
            Factor to increase adaptive epsilon (default is 1.25).
        gamma2 : float, optional
            Factor to decrease adaptive epsilon (default is 0.75).
            
        Returns:
        --------
        arr_par1 : ndarray
            Array of parameter values at each iteration.
        arr_energy : ndarray
            Array of energy values at each iteration.
        arr_var : ndarray
            Array of variance values at each iteration.
        est_energy : float
            Final estimated energy.
        est_variance : float
            Final estimated variance.
        arr_steps : ndarray
            Array of the number of steps taken at each iteration.
        arr_höf : ndarray
            Array of Hoeffding bound values at each iteration.
        arr_höf_adap : ndarray
            Array of adaptive Hoeffding bound values at each iteration.
        arr_grad1 : ndarray
            Array of gradient values at each iteration.
        arr_momentum : ndarray
            Array of momentum values at each iteration.
        arr_epsilon : ndarray
            Array of adaptive epsilon values at each iteration.
    """

    par1 = -1  # Initial parameters
    range_g = 2*np.sum(np.abs(hamiltonian_coeff[1:]))
    E_corr = np.linalg.eigvalsh(h2_op(hamiltonian_coeff))[0]
    max_sample_FINAL = hoeffding_bound(delta= delta,rng=range_g,epsilon=eps_bern)

    '''
    Initialization of Arrays
    '''
    momentum_memory = np.zeros(max_it+1)
    arr_energy = np.zeros(max_it+1)
    arr_var = np.zeros(max_it+1)
    arr_par1 = np.zeros(max_it+1)
    arr_steps = np.zeros(max_it+1,dtype=int) # max_it training steps at most + 1 for final read-out
    arr_höf = np.zeros_like(arr_steps)
    arr_höf_adap = np.zeros_like(arr_steps)
    arr_grad1 = np.zeros(max_it+1)
    arr_momentum = np.zeros(max_it+1)
    arr_max_flag = np.zeros(max_it+1,dtype=bool)
    arr_epsilon = np.zeros(max_it+1)
    # initialize adam optimizers
    adam = AdamOptimizer(par1, alpha=alpha)
    
    momentum = 0

    for i in range(max_it):
        print("Step:",i,"...","Theta: ",par1)
        # set states to |01>
        # Apply circuit on state +/- eps to estimate the gradient
        state_normal      = circ(par1) # just for internal tracking and final read-out
        state_shift_plus  = circ(par1 + eps)
        state_shift_minus = circ(par1 - eps)

        '''
        Loops for EBS algorithm
        '''
        if adaptive_eps:
            if i == 0:
                adap_eps = eps_bern
            elif momentum - prev_momentum >= 0:
                adap_eps = max(adap_eps*gamma1, eps_bern)
            else:
                adap_eps = min(adap_eps*gamma2,2*range_g)
        else:
            adap_eps = eps_bern

        max_sample = hoeffding_bound(delta= delta,rng=range_g,epsilon=adap_eps) # adaptive epsilon also changes the Hoeffding bound

        # Resets EBS every outer loop iteration
        ebs_shift_plus  = EBS(delta=delta, epsilon=adap_eps, range_of_rndvar=range_g, num_groups=3)
        ebs_shift_minus = EBS(delta=delta, epsilon=adap_eps, range_of_rndvar=range_g, num_groups=3)

        '''
        Prepare Samples for energy and gradient estimatation
        '''
        
        energy   = expected_value(state_normal, h2_op(hamiltonian_coeff))    
        variance = expected_value(state_normal, np.linalg.matrix_power(h2_op(hamiltonian_coeff), 2)) - energy**2

        # EBS for Energy at x+ε
        while ebs_shift_plus.cond_check() and 3*ebs_shift_plus.get_numsamples() < max_sample:
            # inner_cond_check() happens already with add_sample()
            ebs_shift_plus.add_sample(h2_measure(state_shift_plus,hamiltonian_coeff))
        energy_shifted_plus = ebs_shift_plus.get_estimate()
        
        # EBS for Energy at x-ε
        while ebs_shift_minus.cond_check() and 3*ebs_shift_minus.get_numsamples() < max_sample:
            ebs_shift_minus.add_sample(h2_measure(state_shift_minus,hamiltonian_coeff))
        energy_shifted_minus = ebs_shift_minus.get_estimate()
        
        # Estimate the Gradient via SPSA method
        grad1 = (energy_shifted_plus-energy_shifted_minus) / (2*eps)

        # Save data
        arr_energy[i]     = energy
        arr_var[i]        = variance
        arr_par1[i]       = par1
        # count steps for epsilon_plus AND epsilon_minus
        arr_steps[i]      = min(3*ebs_shift_plus.get_numsamples(),max_sample) + min(3*ebs_shift_minus.get_numsamples(),max_sample)
        arr_höf[i]        = 2*max_sample_FINAL
        arr_höf_adap[i]   = 2*max_sample
        arr_max_flag[i]   = True # keep these indices for storate later
        arr_grad1[i]      = grad1
        arr_momentum[i]   = momentum
        arr_epsilon[i]    = adap_eps
        
        if np.abs(energy - E_corr) <= eps_bern:
            break
        
        # Adam optimizer
        prev_par = par1
        par1 = adam.updated_alpha(grad1)
        prev_momentum = momentum
        momentum = adam.get_momentum()
    
    # final energy measurement with eps_bern as accuracy
    ebs = EBS(delta=delta, epsilon=eps_bern , range_of_rndvar=range_g, num_groups=3)
    # EBS
    while ebs.cond_check() and 3*ebs.get_numsamples() < max_sample_FINAL:
        ebs.add_sample(h2_measure(state_normal,hamiltonian_coeff))
    est_energy        = ebs.get_estimate()
    est_variance      = ebs.get_var()[-1]    
    arr_steps[i+1]    = min(3*ebs.get_numsamples(),max_sample_FINAL)
    arr_höf[i+1]      = max_sample_FINAL
    arr_höf_adap[i+1] = max_sample_FINAL
    arr_max_flag[i+1] = True # keep the final read-out values as well
    
    # cut-away excessive zeros in numpy arrays:
    arr_par1     = arr_par1[arr_max_flag]
    arr_energy   = arr_energy[arr_max_flag]
    arr_var      = arr_var[arr_max_flag]
    arr_grad1    = arr_grad1[arr_max_flag]
    arr_momentum = arr_momentum[arr_max_flag]
    arr_epsilon  = arr_epsilon[arr_max_flag]
    
    arr_steps    = arr_steps[arr_max_flag]
    arr_höf      = arr_höf[arr_max_flag]
    arr_höf_adap = arr_höf_adap[arr_max_flag]
    
    return arr_par1, arr_energy,arr_var, est_energy, est_variance, arr_steps, arr_höf, arr_höf_adap,arr_grad1,arr_momentum,arr_epsilon