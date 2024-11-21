import numpy as np

"""
This module implements the Empirical Bernstein Algorithm (EBA) with geometric sampling and mid-interval stopping.
-------------------------------------------------------------------------------------------------------------------
Classes:
- Welford: Class for calculating the mean and standard deviation using Welford's method.
- EBS_UnModded: Empirical Bernstein Algorithm with geometric sampling and mid-interval stopping.
- EBS: Empirical Bernstein Algorithm with geometric sampling and mid-interval stopping, with additional parameters.

Functions:
- Hoeffding Bound: Calculates the minimum number of samples needed to achieve a given accuracy and confidence using the Hoeffding bound.
"""

def hoeffding_bound(delta, epsilon, rng):
    """
    Calculate the Hoeffding bound.

    The Hoeffding bound is a measure of the probability that the sum of random variables deviates from its expected value.

    Parameters
    ----------
    delta : float
        The confidence level, typically a small value such as 0.05.
    epsilon : float
        The desired accuracy.
    rng : float
        The range of the random variables.

    Returns
    -------
    int
        The Hoeffding bound.
    """
    return int(0.5*np.log(2/delta)*rng**2/epsilon**2)

class Welford():
    """
    Welford's online algorithm for computing the mean and standard deviation.
    This class implements Welford's algorithm which allows for the mean and 
    standard deviation to be updated incrementally with each new data point.
    Attributes
    ----------
    n : int
        The number of data points seen so far.
    M : float
        The current mean of the data points.
    S : float
        The sum of squared differences from the current mean.
    Methods
    -------
    update(x)
    mean
    std
        Initializes the Welford object.
        Parameters
        ----------
        a_list : list, optional
            A list of initial data points to initialize the algorithm (default is None).
        Parameters
        ----------
        x : float
            The new data point.
        Returns
        -------
        Returns
        -------
        float
            The current mean.
        Returns
        -------
        float
            The current standard deviation.
    """

    def __init__(self, a_list=None):
        self.n = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        """
        Updates the mean and sum of squared differences with a new data point.

        Args:
            x (float): The new data point.

        Returns:
            None
        """
        self.n += 1
        newM = self.M + (x - self.M) / self.n
        newS = self.S + (x - self.M) * (x - newM)
        self.M = newM
        self.S = newS

    @property
    def mean(self):
        """
        Returns the current mean.

        Returns:
            float: The current mean.
        """
        return self.M

    @property
    def std(self):
        """
        Returns the current standard deviation.

        Returns:
            float: The current standard deviation.
        """
        if self.n == 1:
            return 0
        return np.sqrt(self.S / (self.n - 1))

class EBS_UnModded():
    """
    Empirical Bernstein Stopping (EBS) algorithm implementation.
    Algorithm after Mnih et al. 2008, "Empirical Bernstein Stopping".

    Parameters
    ----------
    delta : float, optional
        Confidence level parameter, by default 0.1.
    epsilon : float, optional
        Precision parameter, by default 0.1.
    range_of_rndvar : float, optional
        Range of the random variable, by default 1.
    beta : float, optional
        Growth rate parameter for geometric sampling, by default 1.1.

    Usage
    -------
    >>> ebs = EBS(delta=0.1, epsilon=0.1, range_of_rndvar=1, beta=1.1)
    >>> while ebs.cond_check():
    >>>    sample = np.random.uniform(0,1)
    >>>    ebs.add_sample(sample)
    >>> mean = ebs.get_estimate()
    0.5

    Methods
    -------
    add_sample(sample): Adds a sample to the list of samples and updates the parameters.
    get_ct() : Returns the array of c_t values.
    get_estimate() : Returns the last estimated mean.
    get_mean() : Returns the array of estimated means.
    get_var() : Returns the array of variances. 
    get_numsamples() : Returns the number of samples used.
    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1, beta=1.1):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.p = 1.1
        self.c = self.delta*(self.p-1)/self.p
        self.beta = beta
        self.x = 0
        self.alpha = 0
        self.current_k = 0
        self.current_t = 1
        self.cons = 3/((delta*(self.p-1))/self.p)
        self.welf = Welford()

    def add_sample(self, sample):
        """
        Adds a sample to the list of samples and updates the parameters.

        Parameters:
        - sample (float): The sample value.
        """
        self.samples.append(sample)
        self.sample_sum += sample
        cur_mean = np.divide(self.sample_sum, self.current_t)
        self.running_mean.append(cur_mean)
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))
        self.current_t = self.current_t + 1

        # Inner loop condition check
        self.inner_cond_check()

    def cond_check(self):
        """
        Checks if the EBA should stop or continue.

        Returns:
        - bool: True if EBA should continue, False if EBA should stop.
        """
        if self.current_k == 0:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    def inner_cond_check(self):
        """
        Check if the inner loop condition is satisfied.

        Returns:
        - none
        updates ct if the condition is satisfied
        """
        if self.current_t > np.floor(self.beta**self.current_k):
            self.update_ct()

    def calc_ct(self):
        """
        Calculates the c_t value for a given time t.

        Returns:
        - float: The c_t value.
        """
        return np.sqrt(2*self.running_variance[-1]*self.x/self.current_t)+3*self.range_of_rndvar*self.x/self.current_t

    def update_ct(self):
        """
        Updates the c_t value.
        """
        self.current_k += 1
        self.alpha = np.floor(self.beta**self.current_k) / \
            np.floor(self.beta**(self.current_k-1))
        self.x = -self.alpha*np.log(self.c/(3*(self.current_k**self.p)))
        self.ct.append(self.calc_ct())

    def get_ct(self):
        """
        Returns the array of c_t values.

        Returns:
        - numpy.ndarray: The array of c_t values.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Returns the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Returns the array of estimated means.

        Returns:
        - numpy.ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Returns the array of variances.

        Returns:
        - numpy.ndarray: The array of variances.
        """
        return np.asarray(self.running_variance)

    def get_numsamples(self):
        """
        Returns the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_t

class EBS():
    """
    Empirical Bernstein Stopping (EBS) algorithm implementation.
    Algorithm modified to include an optimized partion of failure probabilities over maxium number of samples given by hoefdings bound.
    While also utilising a constant partion instead of am exponitally descreasing one.

    Parameters
    ----------
    delta : float, optional
        Confidence level parameter, by default 0.1.
    epsilon : float, optional
        Precision parameter, by default 0.1.
    range_of_rndvar : float, optional
        Range of the random variable, by default 1.
    beta : float, optional
        Growth rate parameter for geometric sampling, by default 1.1.

    Usage
    -------
    >>> ebs = EBS(delta=0.1, epsilon=0.1, range_of_rndvar=1, beta=1.1)
    >>> while ebs.cond_check():
    >>>    sample = np.random.uniform(0,1)
    >>>    ebs.add_sample(sample)
    >>> mean = ebs.get_estimate()
    0.5

    Methods
    -------
    add_sample(sample): Adds a sample to the list of samples and updates the parameters.
    get_ct() : Returns the array of c_t values.
    get_estimate() : Returns the last estimated mean.
    get_mean() : Returns the array of estimated means.
    get_var() : Returns the array of variances. 
    get_numsamples() : Returns the number of samples used.
    """
    
    def _get_N_checks(self,Num_max,beta,constant=1):
        """ finds maximal i s.t. \constant * floor(beta^{i-1}) is at most Num_max """
        guess = int(np.rint(np.log(Num_max/constant)/np.log(beta)))
        # if guess is undershot, increase gradually. If not, jump directly to second while-loop (prevent overshooting)
        while np.floor(beta**guess) <= Num_max:
            guess += 1
        while np.floor(beta**guess) > Num_max:
            guess -= 1
        return guess
    
    def _get_k_min(self,beta,N_min=10):
        # we do not want to calculate EBS bound if N_samples < N_min
        # beta^k >= N_min => floor(beta^k) >= N_min
        # choose ceil(log_beta(N_min)) as output
        return int(np.ceil(np.log(N_min)/np.log(beta)))

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1, beta=1.1,p=1.1,num_groups=1,N_min=10):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.p = p
        #self.c = self.delta*(self.p-1)/self.p
        self.beta = beta
        self.x = 0
        self.alpha = 0
        self.k_0 = self._get_k_min(beta,N_min)
        self.num_groups = num_groups
        self.N_max = hoeffding_bound(delta,epsilon,range_of_rndvar)
        N_checks = self._get_N_checks(self.N_max,beta,num_groups)
        if self.k_0 + 1 >= N_checks:
            self.k_0 = 1
            self.current_k = 1
            self.dt  = delta/N_checks
        else:
            self.current_k = self.k_0
            self.dt = delta/(N_checks-self.k_0+1)
        self.current_t = 0
        self.welf = Welford()
        self.log_dt_3 = np.log(self.dt/3)
        self.N_max = self.N_max / num_groups
        
    def reset(self):
        """ Resets all tracked variables and intermediate values """
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.x = 0
        self.alpha = 0
        self.current_k = self.k_0
        self.welf = Welford()
        self.current_t = 0

    def add_sample(self, sample):
        """
        Adds a sample to the list of samples and updates the parameters.

        Parameters:
        - sample (float): The sample value.
        """
        self.sample_sum += sample
        self.current_t += 1
        cur_mean = np.divide(self.sample_sum, self.current_t)
        self.running_mean.append(cur_mean)
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))

        # Inner loop condition check
        self.inner_cond_check()

    def cond_check(self):
        """
        Checks if the EBA should stop or continue.

        Returns:
        - bool: True if EBA should continue, False if EBA should stop.
        """
        if self.current_k == self.k_0:
            return True
        elif self.num_groups * self.current_t >= self.N_max:
            return False
        return self.ct[-1] > self.epsilon

    def inner_cond_check(self):
        """
        Check if the inner loop condition is satisfied.

        Returns:
        - none
        updates ct if the condition is satisfied
        """
        if self.current_t > np.floor(self.beta**self.current_k):
            #print("Updating bounds at N = {} (k = {})".format(self.current_t,self.current_k))
            self.update_ct()

    def calc_ct(self):
        """
        Calculates the c_t value for a given time t.

        Returns:
        - float: The c_t value.
        """
        return np.sqrt(2*self.running_variance[-1]*self.x/self.current_t)+3*self.range_of_rndvar*self.x/self.current_t

    def update_ct(self):
        """
        Updates the c_t value.
        """
        self.current_k += 1
        self.alpha = np.floor(self.beta**self.current_k) / \
            np.floor(self.beta**(self.current_k-1))
        #self.x = -self.alpha*np.log(self.c/(3*(self.current_k**self.p)))
        self.x = -self.alpha*self.log_dt_3
        self.ct.append(self.calc_ct())

    def get_ct(self):
        """
        Returns the array of c_t values.

        Returns:
        - numpy.ndarray: The array of c_t values.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Returns the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Returns the array of estimated means.

        Returns:
        - numpy.ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Returns the array of variances.

        Returns:
        - numpy.ndarray: The array of variances.
        """
        return np.asarray(self.running_variance)

    def get_numsamples(self):
        """
        Returns the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_t