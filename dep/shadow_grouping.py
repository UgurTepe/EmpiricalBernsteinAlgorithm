# code implementation forked from https://gitlab.com/GreschAI/shadowgrouping and mended 

import numpy as np
from qibo import models, gates # checked for version 0.1.7
from os.path import isfile, join
from qiskit.quantum_info import Pauli # checked for version 0.43.1
from qiskit.opflow import PauliOp, SummedOp
from scipy.sparse.linalg import eigsh
from time import time

from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.algorithms import GroundStateEigensolver

char_to_int = {"I":0,"X":1,"Y":2,"Z":3}
int_to_char = {item: key for key,item in char_to_int.items()}

def hit_by(O,P):
    """ Returns whether o is hit by p """
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

def get_pauli_list(molecule,
                   mapping=JordanWignerMapper,
                   basis="sto3g",
                   driver_type=ElectronicStructureDriverType.PYSCF,
                   driver_kwargs = None,
                   verbose = False
                  ):
    """
        Takes a molecule description as input and returns the corresponding qubit Hamiltonian P.
        Here, P[:,1] is the list of coefficients for the Pauli operators in P[:,0], respectively.
        Additional arguments are:
            basis (str):    Basis set for the chemistry calculations. See qiskit documentation for a list of available types.
            driver_type:    Choose a driver type from the list of supported drivers in the qiskit documentation
            mapping:        Fermion-to-qubit mapping. See qiskit documentation for the list of supported mappings
            driver_kwargs:  Optional arguments passed to the chosen driver of type Union[Dict[str, Any], NoneType]
            verbose (bool): Plot the details of the 2nd quantisation and qubit conversion to console
    """
    # init driver
    driver = ElectronicStructureMoleculeDriver(molecule, basis=basis, driver_type=driver_type, driver_kwargs=driver_kwargs)
    es_problem = ElectronicStructureProblem(driver)
    # 2nd quantisation
    second_q_op = es_problem.second_q_ops()
    if verbose:
        print(second_q_op["ElectronicEnergy"])
        print()
    # fermion-to-qubit mapping
    qubit_converter = QubitConverter(mapping())
    qubit_op = qubit_converter.convert(second_q_op["ElectronicEnergy"])
    if verbose:
        print("Qubit Operator")
        print(qubit_op)
        print()
    # qiskit-to-numpy export
    P = []
    for q in qubit_op.to_pauli_op():
        P.append( [str(q.primitive),q.coeff] )
    return np.array(P,dtype=object)


class StateSampler(): 
    """ Convenience class that holds a fixed state of length 2**num_qubits. The latter number is inferred automatically.
        Provides a sampling method that obtains samples from the state in the chosen basis.

        Input:
        - state, numpy array of size 2**N with N = num_qubits. Coefficients are to be specified in the computational basis.
    """
    def __init__(self,state):
        self.state = np.array(state)
        self.num_qubits = int(np.round(np.log2(len(state)),0))
        assert len(self.state) == 2**self.num_qubits, "State size has to be of size 2**N for some integer N."
        self.circuit = models.Circuit(self.num_qubits)
        self.X = [gates.H,gates.I] # use Hadamard gate to switch to +/- basis
        S_dagger = lambda i: gates.U1(i,-np.pi/2)
        self.Y = [S_dagger,gates.H] # use Hadamard + phase gate to switch to +/-i basis
        self.Z = [gates.I,gates.I] # no need to change if already in comp. basis or not measuring at all
        self.I = self.Z
        return

    def sample(self,meas_basis=None,nshots=1):
        """ Draws <nshots> samples from the state.
            If no measurement basis is defined, samples are drawn in the computational basis.

            Inputs:
            - meas_basis as a Pauli string, i.e. a str of length num_qubits containing only X,Y,Z,I.
            - nshots (int) specifying how many samples to draw.

            Returns:
            - samples, numpy array of shape (nshots x  num_qubits).
        """
        c = self.circuit.copy(deep=True)
        if meas_basis is not None:
            assert len(meas_basis)==self.num_qubits, "Measurement basis has to be specified for each qubit."
            c.add([getattr(self,s)[0](i) for i,s in enumerate(meas_basis)])
            c.add([getattr(self,s)[1](i) for i,s in enumerate(meas_basis)])
        c.add(gates.M(*range(self.num_qubits)))
        out = c(initial_state=self.state.copy(),nshots=nshots)
        out = out.samples()
        # mask-out non-measured qubits in <meas-basis>
        out = out * np.array([s!="I" for s in meas_basis],dtype=int)[np.newaxis,:]
        return -2*out + 1 # map {0,1} to {1,-1} outcomes

    def index_to_string(self,index_list):
        """ Helper function that maps a list of Pauli indices to a Pauli string, i.e.
            0 -> I, 1 -> X, 2 -> Y, 3 -> Z
            Returns the Pauli string.
        """
        pauli_string = ""
        for ind in np.array(index_list,dtype=int):
            assert ind in range(4), "Elements of index_list have to be in {0,1,2,3}."
            pauli_string += int_to_char[ind]
        return pauli_string
    
class L1_sampler:
    """ Comparison class that does not reconstruct the Hamiltonian expectation value by its components, but by its relative signs. """

    def __init__(self,observables,weights):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        abs_vals = np.abs(weights)
        #print(len(abs_vals))

        self.obs         = observables
        self.num_obs     = M
        self.num_qubits  = n
        self.w           = weights
        self.prob        = abs_vals / np.sum(abs_vals)
        self.shots       = 0
        self.is_sampling = True
        self.is_adaptive = False

        return

    def reset(self):
        self.shots = 0

    def find_setting(self,num_samples=1):
        self.shots += num_samples
        inds = np.random.choice(self.num_obs,size=(num_samples,),p=self.prob)
        return inds

    def get_Hoeffding_bound(self,epsilon):
        return 2*np.exp(-0.5*epsilon**2*self.shots/np.sum(np.abs(self.w))**2)

    def get_epsilon(self,delta):
        return np.sqrt(2/self.shots*np.log(2/delta)) * np.sum(np.abs(self.w))
    
class Sign_estimator():

    def __init__(self,measurement_scheme,state,offset):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings = 0
        self.num_outcomes = 0

    def reset(self):
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()

    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        inds = self.measurement_scheme.find_setting(num_steps)
        self.setting_inds = np.append((self.setting_inds,inds)) if len(self.setting_inds)>0 else inds
        self.num_settings += num_steps
        return


    def measure(self):
            """ If there are proposed settings in self.settings that have not been measured, do so.
                The internal state of the VQE does not alter by doing so.
            """
            num_meas = self.num_settings - self.num_outcomes
            if num_meas > 0:
                # run all the last prepared measurement settings
                # from the settings list, fetch the unique settings and their respective counts
                recent_settings = self.setting_inds[-num_meas:]            
                outcomes = np.zeros(num_meas,dtype=int)
                for unique,nshots in zip(*np.unique(recent_settings,return_counts=True)):
                    setting = self.measurement_scheme.obs[unique]
                    samples = self.state.sample(meas_basis=self.state.index_to_string(setting),nshots=nshots)
                    outcomes[recent_settings==unique] = np.prod(samples,axis=-1)
                self.outcomes = np.append(self.outcomes,outcomes)
                self.num_outcomes += num_meas
            else:
                print("No more measurements required at the moment. Please propose new setting(s) first.")
            return
    
    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        if self.num_outcomes == 0:
            # if no measurements have been done yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        sgn = np.sign(w)
        norm = np.sum(np.abs(w))
        energy = np.mean(self.outcomes*sgn[self.setting_inds])*norm
        return energy + self.offset
    
    def get_energy_samples(self):
        """ Takes the current outcomes and returns them """
        if self.num_outcomes == 0:
            # if no measurements have been done yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        sgn = np.sign(w)
        norm = np.sum(np.abs(w))
        energy = self.outcomes*sgn[self.setting_inds]*norm
        #print(energy)
        return energy + self.offset
    
class Hamiltonian():
    """ Helper class to turn a list of Pauli operators with accompanying weights into a (sparse) Hamiltonian and diagonalize it.
        Code copied and modified from https://github.com/charleshadfield/adaptiveshadows/blob/main/python/hamiltonian.py
    """

    def __init__(self, weights, observables):
        self.weights = weights
        self.observables = observables

    def SummedOp(self):
        paulis = []
        for P, coeff_P in zip(self.observables,self.weights):
            paulis.append(coeff_P * PauliOp(Pauli(P)))
        return SummedOp(paulis)

    def ground(self, sparse=False):
        if not sparse:
            mat = self.SummedOp().to_matrix()
            evalues, evectors = np.linalg.eigh(mat)
        else:
            mat = self.SummedOp().to_spmatrix()
            evalues, evectors = eigsh(mat, which='SA')
            # SA looks for algebraically small evalues
        index = np.argmin(evalues)
        ground_energy = evalues[index]
        ground_state = evectors[:,index]
        return ground_energy, ground_state

def load_pauli_list(folder_hamiltonian,molecule_name="H2",basis_name="STO3g",encoding="JW",verbose=False,sparse=False,diagonalize=True):
    """ Loads the Pauli operators and the corresponding ground-state energy from the files of
        https://github.com/charleshadfield/adaptiveshadows
        Requires the name of the folder where all the Hamiltonians are stored together with the selection of the
        molecule, basis set and encoding. If verbose is set to True, some elements of the Pauli list are printed to console.
        If sparse is set to True, carries out the numerical diagonalization on a sparse form of the Hamiltonian.
        If diagonalize is set to False, only returns the Pauli decomposition from file and sets all other return values to None.

        Returns the observables, their respective weight, the offset energy, the exact ground-state energy and the state.
    """
    filename = "{}_{}_{}.txt".format(molecule_name,basis_name,encoding)
    full_file_name = join(folder_hamiltonian,filename)
    #assert isfile(full_file_name), "File not found: {}".format(full_file_name)     
    
    # extract Pauli list from file
    data = np.loadtxt('config/H2_STO3g_JW.txt',dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real

    if diagonalize:
        # use Pauli list to create Hamiltonian and diagonalize it afterwards to obtain ground-state
        H = Hamiltonian(weights,paulis)
        E_numerics, state = H.ground(sparse=sparse)
    else:
        E_numerics = None
        state = None

    # Pauli item "III...II" in list should correspond to energy offset
    ind = -1
    identity = "I"*len(paulis[0])
    for i,p in enumerate(paulis):
        if p == identity:
            ind = i
            break
    if ind == -1:
        offset = 0
        obs = paulis
        w = weights
    else:
        offset = weights[ind]
        # erase the corresponding entry in paulis and weights
        obs = np.delete(paulis,ind)
        w = np.delete(weights,ind)
        assert len(obs) == len(paulis) - 1, "Error in line eraser."
        assert len(obs) == len(w), "Both arrays are not of equal length anymore."

    # print some to console
    if verbose:
        print("Offset","\t\t",offset)
        for i, (p, we) in enumerate(zip(obs, w)):
            print(p,"\t",we)
            if i == 9:
                print("\t","...")
                break

    # convert string characters to integers
    observables = np.array([[char_to_int[c] for c in o] for o in obs],dtype=int)

    return observables, w, offset, E_numerics, state

class Measurement_scheme:
    """ Parent class for measurement schemes. Requires
        observables: Array of shape (num_obs x num_qubits) with entries in {0,1,2,3} (the Pauli operators)
        weights:     Array of shape (num_obs) with the corresponding weight in the Hamiltonian decomposition.
                     Array is flattened upon input.
        epsilon:     Absolute error threshold, see child methods for an individual interpretation.
    """
    
    def __init__(self,observables,weights,epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"
        
        self.obs           = observables
        self.num_obs       = M
        self.num_qubits    = n
        self.w             = weights
        self.eps           = epsilon
        self.scheme_params = {"eps": epsilon, "num_obs": M}
        self.N_hits        = np.zeros(M,dtype=int)
        self.is_adaptive   = False # useful default to be given to any child class
        
        return
        
    def find_setting(self):
        pass
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return
        
class Shadow_Grouping(Measurement_scheme):
    """ Grouping method based on weights obtained from classical shadows.
        The next measurement setting p is found as follows: it is initialized as the identity operator.
        Next, we obtain an ordering of the observables in terms of their respective weight_function.
        For each observable o in the ordered list of observables in descending order, it checks qubit-wise commutativity (QWC).
        If so, the qubits in p that fall in the support of o are overwritten by those in o.
        Eventually, the list is either exhausted or p does not contain identity operators anymore.
        The function weight_function takes in the weights,epsilon and the current number of N_hits and is supposed to return an numpy-array of length len(w).
        Instead, weight_function can also be set to None (this is useful for instances where the function is actually never called).
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return         
        
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)

        if verbose:
            print("Checking list of observables.")
        tstart = time()
        for idx in reversed(order):
            o = self.obs[idx]
            if verbose:
                print("Checking",o)
            if hit_by(o,setting):
                non_id = o!=0
                # overwrite those qubits that fall in the support of o
                setting[non_id] = o[non_id]
                if verbose:
                    print("p =",setting)
                # break sequence is case all identities in setting are exhausted
                if np.min(setting) > 0:
                    break
                    
        tend = time()

        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        # further info for comparisons
        info = {}
        return setting, info
    
class Shadow_Grouping_Minimal(Shadow_Grouping):
    """ Grouping method based on weights obtained from classical shadows.
        The next measurement setting p is found as follows: it is initialized as the identity operator.
        Next, we obtain an ordering of the observables in terms of their respective weight_function.
        For each observable o in the ordered list of observables in descending order, it checks qubit-wise commutativity (QWC).
        If so, the qubits in p that fall in the support of o are overwritten by those in o.
        Eventually, the list is either exhausted or p does not contain identity operators anymore.
        The function weight_function takes in the weights,epsilon and the current number of N_hits and is supposed to return an numpy-array of length len(w).
        Instead, weight_function can also be set to None (this is useful for instances where the function is actually never called).
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function,buffer_size=None,allow_overlaps=True):
        super().__init__(observables,weights,epsilon,weight_function)
        self.buffer = {} # key: Pauli string, value: which observables are hit
        self.buffer_max_size = buffer_size
        self.allow_overlaps  = allow_overlaps
        if buffer_size is not None:
            assert isinstance(buffer_size,(int,np.int64)), "<buffer_size> has to be integer or None but was {}.".format(type(buffer_size))
            assert buffer_size > 0, "<buffer_size> has to be positive but was {}.".format(buffer_size)
        return
    
    @property
    def buffer_size(self):
        return len(self.buffer.keys())
    
    def reset(self):
        super().reset()
        self.buffer = {}
        return
    
    def __setting_to_str(self,p):
        out = ""
        for c in p:
            out += int_to_char[c]
        return out
        
    def find_setting(self):
        """ Finds the next measurement setting."""
        fill_buffer = np.min(self.N_hits) == 0 if self.buffer_max_size is None else self.buffer_size < self.buffer_max_size
        if fill_buffer:
            # allocate as in the super-class until all observables have been measured once and fill the buffer
            setting, info = super().find_setting()
            if self.allow_overlaps:
                is_hit = np.array([hit_by(o,setting) for o in self.obs])
            else:
                # mask out those observables that have already been put into a cluster
                is_hit = np.array([hit_by(o,setting) and self.N_hits[i]==1 for i,o in enumerate(self.obs)])
                assert np.sum(is_hit) > 0, "No new observable assigned!"
            self.buffer[self.__setting_to_str(setting)] = is_hit
        else:
            # go through the buffer and pick greedily among settings there
            val, setting = 0.0, ""
            for p, is_hit in self.buffer.items():
                # calculate the sum of hitted weights using the objective function
                bound = np.sum(self.weight_function(self.w,self.eps,self.N_hits)[is_hit])
                if bound > val:
                    val, setting = bound, p
            assert setting != "", "No setting allocated, despite having {} elements in buffer.".format(len(self.buffer))
            self.N_hits += self.buffer[setting]
            setting = [char_to_int[c] for c in setting]
            info = {}
        return setting, info
    
class Bernstein_bound():
    def __init__(self,alpha=1):
        self.alpha = alpha
        assert alpha >= 1, "alpha has to be chosen larger or equal 1, but was {}.".format(alpha)
        return
    
    def get_weights(self,w,eps,N_hits):
        inconf = self.alpha * np.abs(w)
        condition = N_hits != 0
        N = np.sqrt(N_hits[condition])
        Nplus1 = np.sqrt(N_hits[condition] + 1)
        inconf[condition] /= self.alpha*np.sqrt(N*Nplus1)/(Nplus1-N)
        return inconf
    
    def __call__(self):
        return self.get_weights
    
class Energy_estimator():
    """ Convenience class that holds both a measurement scheme and a StateSampler instance.
        The main workflow consists of proposing the next (few) measurement settings and measuring them in the respective bases.
        Furthermore, it tracks all measurement settings and their respective outcomes (of value +/-1 per qubit).
        Based on these values, the current energy estimate can be calculated.
        
        Inputs:
        - measurement_scheme, see class Measurement_Scheme and subclasses for information.
        - state, see class StateSampler.
        - Energy offset (defaults to 0) for the energy estimation.
          This consists of the identity term in the corresponding Hamiltonian decomposition.
    """
    def __init__(self,measurement_scheme,state,offset=0):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        # convenience counters to keep track of measurements settings and respective outcomes
        self.settings_dict = {}
        self.settings_buffer = {}
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.num_settings = 0
        self.num_outcomes = 0
        self.measurement_scheme.reset()
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        
    def reset(self):
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.settings_dict = {}
        self.settings_buffer = {}
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()
        return
    
    def toggle_outcome_tracking(self):
        info = "toggle_outcome_tracking: Tracking of outcomes "
        info += "enabled. They are temporarily stored in self.outcome_dict (dict) once self.measure() is envoked. "
        info += "Each call of self.measure() overwrites any stored data.\n"
        info += "Calling this method again disables the storage of outcomes altogether."
        print(info)
        self.outcome_dict = {}
         
    def clear_outcomes(self):
        self.settings_buffer = self.settings_dict.copy()
        self.running_avgs = np.zeros_like(self.measurement_scheme.w)
        self.running_N    = np.zeros(len(self.running_avgs),dtype=int)
        self.num_outcomes = 0
        return
    
    def __setting_to_str(self,p):
        out = ""
        for c in p:
            out += int_to_char[c]
        return out
    
    def __settings_to_dict(self,settings):
        # run all the last prepared measurement settings
        # from the settings list, fetch the unique settings and their respective counts
        unique_settings, counts = np.unique(settings,axis=0,return_counts=True)
        for setting,nshots in zip(unique_settings,counts):
            paulistring = self.__setting_to_str(setting)
            for diction in (self.settings_dict,self.settings_buffer):
                val = diction.get(paulistring,0)
                diction[paulistring] = nshots + val
        return
        
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        settings = []
        for i in range(num_steps):
            p, _ = self.measurement_scheme.find_setting()
            settings.append(p)
        settings = np.array(settings)
        self.num_settings += num_steps
        self.__settings_to_dict(settings)
        return
    
    def measure(self):
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated. Please allocate measurements first by calling propose_next_settings() first.")
            return
        for setting,reps in self.settings_buffer.items():
            # measure <reps> times in <setting>
            samples = self.state.sample(meas_basis=setting,nshots=reps)
            # write into running_avgs
            for i,o in enumerate(self.measurement_scheme.obs):
                if not hit_by(o,[char_to_int[c] for c in setting]):
                    continue
                mask = np.zeros(samples.shape,dtype=int)
                mask += (o == 0)[np.newaxis,:]
                temp = samples.copy()
                temp[mask.astype(bool)] = 1 # set to 1 if outside the support of the respective hit observable to mask it out
                self.running_avgs[i] = ( self.running_avgs[i]*self.running_N[i] + np.prod(temp,axis=1).sum() ) / (self.running_N[i] + reps)
                self.running_N[i] += reps
        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
        return
    
    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        energy = np.sum(self.measurement_scheme.w*self.running_avgs)
        return energy + self.offset
    
class EBS_naive_estimator():
    
    def __init__(self,method,state,offset=0):
        #assert not method.allow_overlaps, "EBS requires the method not to allow for overlaps (allow_overlaps should be False"
        self.measurement_scheme = method
        self.state = state
        self.offset = offset
        
        estimator = Energy_estimator(method,state,offset)
        estimator.propose_next_settings(method.num_obs)
        assert np.min(estimator.measurement_scheme.N_hits) > 0, "Not enough settings provided to infer all groups"
        self.buffer = estimator.measurement_scheme.buffer.copy()
        self.groups = list(estimator.measurement_scheme.buffer.keys())
        assert np.min(np.array([val for val in self.buffer.values()]).sum(axis=0)) > 0, "Not all observable contained in groups"
        self.num_groups = len(self.groups)
        self.measurement_scheme.reset()
        
        self.num_samples = 0
        
    def reset(self):
        self.measurement_scheme.reset()
        self.num_samples = 0
        
    def get_samples(self,num_samples=1):
        outcome_avg = np.zeros((self.measurement_scheme.num_obs,num_samples)) # shape M X N
        for setting,is_hit in self.buffer.items():
            outcomes = self.state.sample(setting,num_samples) # shape N
            for obs_idx in np.arange(self.measurement_scheme.num_obs)[is_hit]:
                # go through the hit observables only
                mask = np.zeros(outcomes.shape,dtype=int)
                mask += (self.measurement_scheme.obs[obs_idx] == 0)[np.newaxis,:]
                temp = outcomes.copy()
                temp[mask.astype(bool)] = 1 # set to 1 if outside the support of the respective hit observable to mask it out
                outcome_avg[obs_idx] += np.prod(temp,axis=1)
            self.measurement_scheme.N_hits += self.buffer[setting].astype(int)*num_samples
        # if one observable is contained in several groups, average the corresponding outcome over the number of eligible groups
        # by design of the measurement_scheme, there is at least one eligible group
        outcome_avg = outcome_avg / np.array([val for val in self.buffer.values()]).sum(axis=0)[:,np.newaxis]
        self.num_samples += num_samples
        energy = np.sum(self.measurement_scheme.w[:,np.newaxis] * outcome_avg,axis=0) + self.offset
        return energy # shape N
    
def get_estimator(w,observables,offset,state):
    """ Create groups following the ShadowGrouping protocol based on the Hamiltonian in Pauli decomposition, i.e., H = (w,observables) + offset.
        Automatically builds the groups and measures each of them repeatedly given a quantum state <state>
        In consequence, each one of the energy samples is comprised of <N_groups> measurements of <state>.
        
        Returns instance of energy estimator class
    """
    get_extremes = lambda x: (np.min(x), np.max(x))
    alpha, temp = get_extremes(np.abs(w))
    alpha += temp/alpha
    method = Shadow_Grouping_Minimal(observables,w,0.1,Bernstein_bound(alpha)(),allow_overlaps=False)
    sampler = StateSampler(state)
    estimator = EBS_naive_estimator(method,sampler,offset)
    return estimator