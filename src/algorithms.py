import numpy as np
'''
                A collection of various stopping algorithm.
 ------------ ------------ Relative Accuracy  ------------ ------------
Bernstein based:
    ebs_simple   : basic ebs algorithm
    ebs_dual     : upper and lower bound version, improved ver. of ebs_simple
Höffding based:
    nas          : stopping algorithm based on Höffding's inequality
 ------------ ------------ Absolute Accuracy  ------------ ------------
Bernstein based:
    eba_simple   : basic ebs algorithm
    eba_geo      : geometric sampling, improved ver. eba
    eba_geo_marg : geometric samp. + mid-interval stopping, improved ver. eba_geo
Höffding based:
    nas_abs      : stopping algorithm based on Höffding's inequality
'''


def hoeffding_bound(delta, epsilon, rng):
    '''
    Hoeffding bound solved for t_min. Calculates the minimum number of samples needed to achieve a given accuracy and confidence.
    '''
    return 0.5*np.log(2/delta)*rng**2/epsilon**2


class Welford():
    """
    Class for calculating the mean and standard deviation using the Welford's method.

    Attributes:
        n (int): The number of data points.
        M (float): The current mean.
        S (float): The current sum of squared differences from the mean.

    Methods:
        update(x): Updates the mean and sum of squared differences with a new data point.
        mean: Returns the current mean.
        std: Returns the current standard deviation.
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


class ebs_simple():
    """
    Empirical Bernstein Algorithm (EBS) implementation.

    Parameters:
    - delta (float): The confidence parameter. Default is 0.1.
    - epsilon (float): Accuracy parameter. Default is 0.1.
    - range_of_rndvar (float): The range of the random variable. Default is 1.

    Attributes:
    - delta (float): The confidence parameter.
    - epsilon (float): Accuracy parameter.
    - range_of_rndvar (float): The range of the random variable.
    - samples (list): List to store the samples.
    - running_mean (list): List to store the running mean.
    - sample_sum (float): The sum of all the samples.
    - running_variance (list): List to store the running variance.
    - ct (list): List to store the ct values.
    - p (float): The p value used in ct calculation.
    - current_step (int): The current step/iteration.
    - cons (float): The constant used in ct calculation.
    - welf (Welford): Welford object for calculating running variance.

    Methods:
    - inner_cond_check(): Placeholder method.
    - add_sample(sample): Adds a sample to the algorithm and updates the parameters.
    - cond_check(): Checks if the algorithm should stop or continue.
    - calc_ct(time): Calculates the ct value for a given time.
    - get_ct(): Returns the array of ct values.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the array of estimated means.
    - get_var(): Returns the array of variances.
    - get_step(): Returns the current step/iteration.
    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        """
        Initialize the EBS algorithm with the given parameters.

        Parameters:
        - delta (float): The confidence parameter. Default is 0.1.
        - epsilon (float): Accuracy parameter. Default is 0.1.
        - range_of_rndvar (float): The range of the random variable. Default is 1.
        """
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = []
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.p = 1.1
        self.current_step = 1
        self.cons = 3/((delta*(self.p-1))/self.p)
        self.welf = Welford()

    def inner_cond_check(self):
        pass

    def add_sample(self, sample):
        """
        Add a sample to the algorithm and update the parameters.

        Parameters:
        - sample: The sample to be added.
        """
        self.samples.append(sample)
        self.sample_sum += sample
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))
        self.ct.append(self.calc_ct(self.current_step))
        self.current_step += 1

    def cond_check(self):
        """
        Check if the algorithm should stop or continue.

        Returns:
        - bool: True if the algorithm should continue, False if it should stop.
        """
        if self.current_step == 1:
            return True
        if self.ct[-1] > self.epsilon * self.running_mean[-1]:
            return True
        else:
            return False

    def calc_ct(self, time):
        """
        Calculate the ct bound value for a given time t.

        Parameters:
        - time (int): The time value.

        Returns:
        - float: The calculated ct value.
        """
        ln_constant = np.log(self.cons, dtype=np.float64) / time
        ln_vari = self.p * np.log(time, dtype=np.float64) / time
        ln_compl = (ln_constant + ln_vari)
        result = (
            np.sqrt(2 * self.running_variance[-1] * ln_compl) + (3 * self.range_of_rndvar * ln_compl))
        return result

    def get_ct(self):
        """
        Get the array of ct values.

        Returns:
        - numpy.ndarray: The array of ct values.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Get the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Get the array of estimated means.

        Returns:
        - numpy.ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Get the array of variances.

        Returns:
        - numpy.ndarray: The array of variances.
        """
        return np.asarray(self.running_variance)

    def get_step(self):
        """
        Get the current step/iteration.

        Returns:
        - int: The current step/iteration.
        """
        return self.current_step


class ebs_dual():
    """
    Empirical Bernstein Algorithm (EBS) class.

    This class implements the Empirical Bernstein Algorithm, which is used for estimating means and variances
    of a sequence of samples. It provides methods for adding samples, calculating bounds, and retrieving
    estimated means and variances.

    Parameters:
    - delta (float): The confidence parameter for the algorithm. Default is 0.1.
    - epsilon (float): The tolerance parameter for the algorithm. Default is 0.1.
    - range_of_rndvar (float): The range of the random variable. Default is 1.

    Attributes:
    - delta (float): The confidence parameter for the algorithm.
    - epsilon (float): The tolerance parameter for the algorithm.
    - range_of_rndvar (float): The range of the random variable.
    - samples (list): The list of samples.
    - running_mean (list): The list of running means.
    - sample_sum (float): The sum of all samples.
    - running_variance (list): The list of running variances.
    - ct (list): The list of c_t values.
    - low_bound (float): The lower bound of the estimated mean.
    - upper_bound (float): The upper bound of the estimated mean.
    - low_bound_max (float): The maximum lower bound encountered so far.
    - upper_bound_min (float): The minimum upper bound encountered so far.
    - p (float): The parameter used in calculating c_t.
    - current_step (int): The current step/iteration of the algorithm.
    - cons (float): The constant used in calculating c_t.
    - welf (Welford): An instance of the Welford class used for calculating running variance.

    Methods:
    - inner_cond_check(): Placeholder method for inner conditional check.
    - add_sample(sample): Adds a sample to the algorithm and updates all the parameters.
    - cond_check(): Checks if the algorithm should stop or continue.
    - calc_ct(time): Calculates the c_t value for a given time.
    - get_ct(): Returns the list of c_t values.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the list of estimated means.
    - get_var(): Returns the list of variances.
    - get_step(): Returns the current step/iteration.

    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.low_bound = 0
        self.upper_bound = 0
        self.low_bound_max = 0
        self.upper_bound_min = 999
        self.p = 1.1
        self.current_step = 1
        self.cons = 3/((delta*(self.p-1))/self.p)
        self.welf = Welford()

    def inner_cond_check(self):
        """
        Placeholder method for inner conditional check.
        """
        pass

    def add_sample(self, sample):
        """
        Adds a sample to the algorithm and updates all the parameters.

        Parameters:
        - sample (float): The sample to be added.

        """
        # Insert new sample
        self.samples.append(sample)

        # cummulative sum
        self.sample_sum += sample

        # Calculates the running mean efficiently with sample_sum
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)

        # Running variance
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))

        # Update ct
        self.ct.append(self.calc_ct(self.current_step))

        # Update lower and upper bounds
        self.low_bound = np.abs(self.running_mean[-1])-self.get_ct()[-1]
        self.upper_bound = np.abs(self.running_mean[-1])+self.get_ct()[-1]

        if self.low_bound > self.low_bound_max:
            self.low_bound_max = self.low_bound
        if self.upper_bound < self.upper_bound_min:
            self.upper_bound_min = self.upper_bound

        # Update current step
        self.current_step = self.current_step + 1

    def cond_check(self):
        """
        Checks if the algorithm should stop or continue.

        Returns:
        - bool: True if the algorithm should continue, False if it should stop.

        """
        if self.current_step == 1:
            return True
        if self.low_bound_max*(1+self.epsilon) < self.upper_bound_min*(1-self.epsilon):
            return True
        else:
            return False

    def calc_ct(self, time):
        """
        Calculates the c_t value for a given time.

        Parameters:
        - time (int): The time step.

        Returns:
        - float: The c_t value.

        """
        ln_constant = np.log(self.cons, dtype=np.float64)/time
        ln_vari = self.p*np.log(time, dtype=np.float64)/time
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def get_ct(self):
        """
        Returns the list of c_t values.

        Returns:
        - numpy.ndarray: The array of c_t values.

        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Returns the latest estimated mean.

        Returns:
        - float: The estimated mean.

        """
        return np.sign(self.running_mean[-1])*0.5*(self.low_bound_max*(1+self.epsilon) + self.upper_bound_min*(1-self.epsilon))

    def get_mean(self):
        """
        Returns the list of estimated means.

        Returns:
        - numpy.ndarray: The array of estimated means.

        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Returns the list of variances.

        Returns:
        - numpy.ndarray: The array of variances.

        """
        return np.asarray(self.running_variance)

    def get_step(self):
        """
        Returns the current step/iteration.

        Returns:
        - int: The current step/iteration.

        """
        return self.current_step


class nas():
    """
    Class representing the NAS (Nonmonotonic Adaptive Sampling) algorithm.

    Parameters:
    - delta (float): The confidence parameter. Default is 0.1.
    - epsilon (float): The precision parameter. Default is 0.1.
    - range_of_rndvar (int): The range of the random variable. Default is 1.

    Attributes:
    - delta (float): The confidence parameter.
    - epsilon (float): The precision parameter.
    - range_of_rndvar (int): The range of the random variable.
    - samples (list): List to store the samples.
    - running_mean (list): List to store the running mean.
    - sample_sum (float): The sum of all the samples.
    - running_variance (list): List to store the running variance.
    - ct (list): List to store the values of c_t.
    - current_step (int): The current step/iteration.

    Methods:
    - inner_cond_check(): Placeholder method.
    - add_sample(sample): Adds a sample to the algorithm and updates the parameters.
    - cond_check(): Checks if the algorithm should stop or continue.
    - calc_ct(time): Calculates the value of c_t for a given time.
    - get_ct(): Returns the values of c_t.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the array of estimated means.
    - get_step(): Returns the current step/iteration.
    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        """
        Initialize the NAS algorithm with the given parameters.

        Parameters:
        - delta (float): The confidence parameter. Default is 0.1.
        - epsilon (float): The precision parameter. Default is 0.1.
        - range_of_rndvar (int): The range of the random variable. Default is 1.
        """
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = []
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.current_step = 1

    def inner_cond_check(self):
        """
        Placeholder method for inner conditional check.
        """
        pass

    def add_sample(self, sample):
        """
        Add a sample to the algorithm and update the parameters.

        Parameters:
        - sample: The sample to be added.
        """
        self.samples.append(sample)
        self.sample_sum += sample
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)
        self.ct.append(self.calc_ct(self.current_step))
        self.current_step += 1

    def cond_check(self):
        """
        Check if the algorithm should stop or continue.

        Returns:
        - bool: True if the algorithm should stop, False otherwise.
        """
        if self.current_step == 1:
            return True
        if np.abs(self.running_mean[-1]) <= self.ct[-1] * (1 + (1 / self.epsilon)):
            return True
        else:
            return False

    def calc_ct(self, time):
        """
        Calculate the value of c_t for a given time.

        Parameters:
        - time (int): The time/step for which to calculate c_t.

        Returns:
        - float: The value of c_t.
        """
        dt = (self.current_step * (self.current_step + 1)) / self.delta
        result = self.range_of_rndvar * \
            np.sqrt(np.log(dt) / (2 * self.current_step))
        return result

    def get_ct(self):
        """
        Get the values of c_t.

        Returns:
        - ndarray: The values of c_t.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Get the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Get the array of estimated means.

        Returns:
        - ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_step(self):
        """
        Get the current step/iteration.

        Returns:
        - int: The current step/iteration.
        """
        return self.current_step


class eba_simple():
    """
    Empirical Bernstein Algorithm (EBA) class.

    Args:
        delta (float): Confidence parameter (default: 0.1).
        epsilon (float): Accuracy (default: 0.1).
        range_of_rndvar (float): Range of the random variable (default: 1).

    Attributes:
        delta (float): Confidence parameter.
        epsilon (float): Accuracy.
        range_of_rndvar (float): Range of the random variable.
        samples (list): List of samples.
        running_mean (list): List of running means.
        sample_sum (float): Sum of all samples.
        running_variance (list): List of running variances.
        ct (list): List of c_t values.
        p (float): Constant parameter.
        current_step (int): Current step/iteration.
        cons (float): Constant value used in c_t calculation.
        welf (Welford): Welford object for calculating running variance.

    Methods:
        inner_cond_check(): Placeholder method.
        add_sample(sample): Adds a sample to the algorithm and updates parameters.
        cond_check(): Checks if the algorithm should stop.
        calc_ct(): Calculates the c_t value.
        update_ct(): Updates the ct list.
        get_ct(): Returns the ct list.
        get_estimate(): Returns the latest estimated mean.
        get_mean(): Returns the array of estimated means.
        get_var(): Returns the array of variances.
        get_step(): Returns the current step/iteration.
    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.p = 1.1
        self.current_step = 1
        self.cons = 3/((delta*(self.p-1))/self.p)
        self.welf = Welford()

    def inner_cond_check(self):
        """
        Placeholder method for inner conditional check.
        """
        pass

    def add_sample(self, sample):
        """
        Adds a sample to the algorithm and updates the parameters.

        Args:
            sample (float): The sample to be added.
        """
        # Insert new sample
        self.samples.append(sample)

        # cummulative sum
        self.sample_sum += sample

        # Calculates the running mean efficiently with sample_sum
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)

        # Running variance
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))

        # Update ct
        self.update_ct()

        # Update current step
        self.current_step = self.current_step + 1

    def cond_check(self):
        """
        Checks if the algorithm should stop.

        Returns:
            bool: True if the algorithm should continue, False otherwise.
        """
        if self.current_step == 1 or self.current_step == 2:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    def calc_ct(self):
        """
        Calculates the c_t value.

        Returns:
            float: The calculated c_t value.
        """
        ln_constant = np.log(self.cons, dtype=np.float64)/self.current_step
        ln_vari = self.p*np.log(self.current_step,
                                dtype=np.float64)/self.current_step
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def update_ct(self):
        """
        Updates the ct list.
        """
        # Update ct
        self.ct.append(self.calc_ct())

    def get_ct(self):
        """
        Returns the ct list.

        Returns:
            numpy.ndarray: The ct list.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Returns the latest estimated mean.

        Returns:
            float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Returns the array of estimated means.

        Returns:
            numpy.ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Returns the array of variances.

        Returns:
            numpy.ndarray: The array of variances.
        """
        return np.asarray(self.running_variance)

    def get_step(self):
        """
        Returns the current step/iteration.

        Returns:
            int: The current step/iteration.
        """
        return self.current_step


class nas_abs():
    """
    Class representing the NAS-ABS algorithm.

    Parameters:
    - delta (float): The confidence parameter for the algorithm. Default is 0.1.
    - epsilon (float): Accuracy for the algorithm. Default is 0.1.
    - range_of_rndvar (float): The range of the random variable. Default is 1.

    Attributes:
    - delta (float): The confidence parameter for the algorithm.
    - epsilon (float): Accuracy for the algorithm.
    - range_of_rndvar (float): The range of the random variable.
    - samples (list): List of samples collected.
    - running_mean (list): List of running means calculated.
    - sample_sum (float): Cumulative sum of the samples.
    - running_variance (list): List of running variances calculated.
    - ct (list): List of c_t values calculated.
    - current_step (int): Current iteration/step of the algorithm.

    Methods:
    - inner_cond_check(): Placeholder method for inner conditional check.
    - add_sample(sample): Adds a sample to the algorithm and updates the parameters.
    - cond_check(): Checks if the algorithm should stop or continue.
    - calc_ct(time): Calculates the c_t value for a given time.
    - get_ct(): Returns the array of c_t values.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the array of estimated means.
    - get_step(): Returns the current iteration/step.
    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        """
        Initialize the NAS-ABS algorithm with the given parameters.

        Parameters:
        - delta (float): The confidence parameter for the algorithm. Default is 0.1.
        - epsilon (float): Accuracy threshold for the algorithm. Default is 0.1.
        - range_of_rndvar (float): The range of the random variable. Default is 1.
        """
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = []
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.current_step = 1

    def inner_cond_check(self):
        """
        Placeholder method for inner conditional check.
        """
        pass

    def add_sample(self, sample):
        """
        Add a sample to the algorithm and update the parameters.

        Parameters:
        - sample (float): The sample to be added.
        """
        self.samples.append(sample)
        self.sample_sum += sample
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)
        self.ct.append(self.calc_ct(self.current_step))
        self.current_step += 1

    def cond_check(self):
        """
        Check if the algorithm should stop or continue.

        Returns:
        - bool: True if the algorithm should continue, False if it should stop.
        """
        if self.current_step == 1:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    def calc_ct(self, time):
        """
        Calculate the c_t value for a given time.

        Parameters:
        - time (int): The time step.

        Returns:
        - float: The calculated c_t value.
        """
        dt = (self.current_step * (self.current_step + 1)) / self.delta
        result = self.range_of_rndvar * \
            np.sqrt(np.log(dt) / (2 * self.current_step))
        return result

    def get_ct(self):
        """
        Get the array of c_t values.

        Returns:
        - numpy.ndarray: The array of c_t values.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Get the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Get the array of estimated means.

        Returns:
        - numpy.ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_step(self):
        """
        Get the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_step


class eba_geo():
    """
    Empirical Bernstein Algorithm (EBA) for geometrically decreasing step sizes.

    Parameters:
    - delta (float): The confidence parameter. Default is 0.1.
    - epsilon (float): Accuracy threshold. Default is 0.1.
    - range_of_rndvar (float): The range of the random variable. Default is 1.
    - beta (float): The geometric decay factor. Default is 1.1.

    Attributes:
    - delta (float): The confidence parameter.
    - epsilon (float): Accuracy threshold.
    - range_of_rndvar (float): The range of the random variable.
    - samples (list): List of samples.
    - running_mean (list): List of running means.
    - sample_sum (float): Sum of all samples.
    - running_variance (list): List of running variances.
    - ct (list): List of c_t values.
    - p (float): The power parameter.
    - beta (float): The geometric decay factor.
    - current_k (int): Current value of k.
    - current_t (int): Current value of t.
    - cons (float): Constant value used in c_t calculation.
    - welf (Welford): Welford object for calculating running variance.

    Methods:
    - add_sample(sample): Adds a sample to the algorithm and updates the parameters.
    - cond_check(): Checks if the EBA algorithm should stop.
    - inner_cond_check(): Checks if the inner loop condition is satisfied.
    - calc_ct(): Calculates the value of c_t for a given time t.
    - update_ct(): Updates the value of c_t.
    - get_ct(): Returns the array of c_t values.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the array of estimated means.
    - get_var(): Returns the array of variances.
    - get_step(): Returns the current iteration/step.
    """

    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1, beta=1.1):
        """
        Initialize the EBA algorithm with the given parameters.

        Parameters:
        - delta (float): The confidence parameter. Default is 0.1.
        - epsilon (float): Accuracy threshold. Default is 0.1.
        - range_of_rndvar (float): The range of the random variable. Default is 1.
        - beta (float): The geometric decay factor. Default is 1.1.
        """
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.p = 1.1
        self.beta = beta
        self.current_k = 0
        self.current_t = 1
        self.cons = 3/((delta*(self.p-1))/self.p)
        self.welf = Welford()

    def add_sample(self, sample):
        """
        Add a sample to the algorithm and update the parameters.

        Parameters:
        - sample: The sample to be added.
        """

        # Insert new sample
        self.samples.append(sample)

        # cummulative sum
        self.sample_sum += sample

        # Calculates the running mean efficiently with sample_sum
        cur_mean = np.divide(self.sample_sum, self.current_t)
        self.running_mean.append(cur_mean)

        # Running variance
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))
        # Update current step
        self.current_t = self.current_t + 1

    def cond_check(self):
        """
        Check if the EBA algorithm should stop.

        Returns:
        - bool: True if the algorithm should continue, False otherwise.
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
        Calculate the value of c_t for a given time t.

        Returns:
        - float: The value of c_t.
        """
        ln_constant = np.log(self.cons, dtype=np.float64)/self.current_t
        ln_vari = self.p*np.log(self.current_k,
                                dtype=np.float64)/self.current_t
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def update_ct(self):
        """
        Update the value of c_t.
        """
        # Update ct
        self.current_k += 1
        self.ct.append(self.calc_ct())

    def get_ct(self):
        """
        Get the array of c_t values.

        Returns:
        - ndarray: Array of c_t values.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Get the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Get the array of estimated means.

        Returns:
        - ndarray: Array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Get the array of variances.

        Returns:
        - ndarray: Array of variances.
        """
        return np.asarray(self.running_variance)

    def get_step(self):
        """
        Get the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_t


class eba_geo_marg():
    """
    Empirical Bernstein Algorithm with Geometric sampling and mid-interval stopping.

    This class implements the Empirical Bernstein Algorithm (EBA) with geometric sampling and mid-interval stopping.
    It provides methods to add samples, update parameters, and retrieve estimates of mean, variance, and more.

    Parameters:
    - delta (float): The confidence parameter. Default is 0.1.
    - epsilon (float): Accuracy threshold. Default is 0.1.
    - range_of_rndvar (float): The range of the random variable. Default is 1.
    - beta (float): batch sampling parameter. Default is 1.1.

    Attributes:
    - delta (float): The confidence parameter.
    - epsilon (float): Accuracy threshold.
    - range_of_rndvar (float): The range of the random variable.
    - samples (list): The list of samples.
    - running_mean (list): The list of running means.
    - sample_sum (float): The sum of all samples.
    - running_variance (list): The list of running variances.
    - ct (list): The list of c_t values.
    - p (float): The parameter p.
    - c (float): The parameter c.
    - beta (float): batch sampling parameter.
    - x (float): The parameter x.
    - alpha (float): The parameter alpha.
    - current_k (int): The current k value.
    - current_t (int): The current t value.
    - cons (float): The constant value.
    - welf (Welford): The Welford object for calculating running variance.

    Methods:
    - add_sample(sample): Adds a sample to the list of samples and updates the parameters.
    - cond_check(): Checks if the EBA should stop or continue.
    - inner_cond_check(): Checks if the inner loop condition is met.
    - calc_ct(): Calculates the c_t value for a given time t.
    - update_ct(): Updates the c_t value.
    - get_ct(): Returns the array of c_t values.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the array of estimated means.
    - get_var(): Returns the array of variances.
    - get_step(): Returns the current iteration/step.
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

    def get_step(self):
        """
        Returns the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_t

class eba_mod():
    """
    Empirical Bernstein Algorithm with Geometric sampling and mid-interval stopping.

    This class implements the Empirical Bernstein Algorithm (EBA) with geometric sampling and mid-interval stopping.
    It provides methods to add samples, update parameters, and retrieve estimates of mean, variance, and more.

    Parameters:
    - delta (float): The confidence parameter. Default is 0.1.
    - epsilon (float): Accuracy threshold. Default is 0.1.
    - range_of_rndvar (float): The range of the random variable. Default is 1.
    - beta (float): batch sampling parameter. Default is 1.1.

    Attributes:
    - delta (float): The confidence parameter.
    - epsilon (float): Accuracy threshold.
    - range_of_rndvar (float): The range of the random variable.
    - samples (list): The list of samples.
    - running_mean (list): The list of running means.
    - sample_sum (float): The sum of all samples.
    - running_variance (list): The list of running variances.
    - ct (list): The list of c_t values.
    - p (float): The parameter p.
    - c (float): The parameter c.
    - beta (float): batch sampling parameter.
    - x (float): The parameter x.
    - alpha (float): The parameter alpha.
    - current_k (int): The current k value.
    - current_t (int): The current t value.
    - cons (float): The constant value.
    - welf (Welford): The Welford object for calculating running variance.

    Methods:
    - add_sample(sample): Adds a sample to the list of samples and updates the parameters.
    - cond_check(): Checks if the EBA should stop or continue.
    - inner_cond_check(): Checks if the inner loop condition is met.
    - calc_ct(): Calculates the c_t value for a given time t.
    - update_ct(): Updates the c_t value.
    - get_ct(): Returns the array of c_t values.
    - get_estimate(): Returns the latest estimated mean.
    - get_mean(): Returns the array of estimated means.
    - get_var(): Returns the array of variances.
    - get_step(): Returns the current iteration/step.
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
        self.no_check_count = 15
        self.check_counter = 0

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
            self.check_counter += 1
            if self.no_check_count <= self.check_counter or self.check_counter == 0:
                self.update_ct()
            else:
                self.ct.append(10)

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

    def get_step(self):
        """
        Returns the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_t
