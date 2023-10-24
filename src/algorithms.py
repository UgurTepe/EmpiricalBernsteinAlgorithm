import numpy as np

# Welford method to get the running standard deviation and the running mean
def hoeffding_bound(delta,epsilon,rng):
    return 0.5*np.log(2/delta)*rng**2/epsilon**2

class Welford():
    def __init__(self, a_list=None):
        self.n = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        self.n += 1
        newM = self.M + (x - self.M) / self.n
        newS = self.S + (x - self.M) * (x - newM)
        self.M = newM
        self.S = newS

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        if self.n == 1:
            return 0
        return np.sqrt(self.S / (self.n - 1))


class ebs_simple():
    # Initlialize Bernstein with epsilon and delta
    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
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

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
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

        # Update current step
        self.current_step = self.current_step + 1

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante
    def cond_check(self):
        if self.current_step == 1:
            return True
        if self.ct[-1] > self.epsilon*self.running_mean[-1]:
            return True
        else:
            return False

    # Just a function to calculate c_t for a given time t
    def calc_ct(self, time):
        ln_constant = np.log(self.cons, dtype=np.float64)/time
        ln_vari = self.p*np.log(time, dtype=np.float64)/time
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return self.running_mean[-1]

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Should return the array of the variances
    def get_var(self):
        return np.asarray(self.running_variance)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_step


class ebs():
    # Initlialize Bernstein with epsilon and delta
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

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
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

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante
    def cond_check(self):
        if self.current_step == 1:
            return True
        if self.low_bound_max*(1+self.epsilon) < self.upper_bound_min*(1-self.epsilon):
            return True
        else:
            return False

    # Just a function to calculate c_t for a given time t
    def calc_ct(self, time):
        ln_constant = np.log(self.cons, dtype=np.float64)/time
        ln_vari = self.p*np.log(time, dtype=np.float64)/time
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return np.sign(self.running_mean[-1])*0.5*(self.low_bound_max*(1+self.epsilon) + self.upper_bound_min*(1-self.epsilon))

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Should return the array of the variances
    def get_var(self):
        return np.asarray(self.running_variance)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_step


class nas():
    # Initlialize Bernstein with epsilon and delta
    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = []
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.current_step = 1

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
        # Insert new sample
        self.samples.append(sample)

        # cummulative sum
        self.sample_sum += sample

        # Calculates the running mean efficiently with sample_sum
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)

        # Update ct
        self.ct.append(self.calc_ct(self.current_step))

        # Update current step
        self.current_step = self.current_step + 1

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante
    def cond_check(self):
        if self.current_step == 1:
            return True
        if np.abs(self.running_mean[-1]) <= self.ct[-1]*(1+(1/self.epsilon)):
            return True
        else:
            return False

    # Just a function to calculate c_t for a given time t
    def calc_ct(self, time):
        dt = (self.current_step*(self.current_step+1))/self.delta
        result = self.range_of_rndvar*np.sqrt(np.log(dt)/(2*self.current_step))
        return result

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return self.running_mean[-1]

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_step


class eba():
    # Initlialize Bernstein with epsilon and delta
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

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
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

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante
    def cond_check(self):
        if self.current_step == 1 or self.current_step == 2:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    # Just a function to calculate c_t for a given time t
    def calc_ct(self, time):
        ln_constant = np.log(self.cons, dtype=np.float64)/time
        ln_vari = self.p*np.log(time, dtype=np.float64)/time
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def update_ct(self):
        # Update ct
        self.ct.append(self.calc_ct(self.current_step))

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return self.running_mean[-1]

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Should return the array of the variances
    def get_var(self):
        return np.asarray(self.running_variance)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_step


class nas_abs():
    # Initlialize Bernstein with epsilon and delta
    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.running_mean = []
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        self.current_step = 1

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
        # Insert new sample
        self.samples.append(sample)

        # cummulative sum
        self.sample_sum += sample

        # Calculates the running mean efficiently with sample_sum
        cur_mean = np.divide(self.sample_sum, self.current_step)
        self.running_mean.append(cur_mean)

        # Update ct
        self.ct.append(self.calc_ct(self.current_step))

        # Update current step
        self.current_step = self.current_step + 1

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante
    def cond_check(self):
        if self.current_step == 1:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    # Just a function to calculate c_t for a given time t
    def calc_ct(self, time):
        dt = (self.current_step*(self.current_step+1))/self.delta
        result = self.range_of_rndvar*np.sqrt(np.log(dt)/(2*self.current_step))
        return result

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return self.running_mean[-1]

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_step


class eba_geo():
    # Initlialize Bernstein with epsilon and delta
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
        self.beta = beta
        self.current_k = 0
        self.current_t = 1
        self.cons = 3/((delta*(self.p-1))/self.p)
        self.welf = Welford()

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
        # Update current step
        self.current_t = self.current_t + 1

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

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante

    def cond_check(self):
        if self.current_k == 0:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    def inner_cond_check(self):
        return self.current_t > np.floor(self.beta**self.current_k)

    # Just a function to calculate c_t for a given time t
    def calc_ct(self):
        ln_constant = np.log(self.cons, dtype=np.float64)/self.current_t
        ln_vari = self.p*np.log(self.current_k,
                                dtype=np.float64)/self.current_t
        ln_compl = (ln_constant+ln_vari)
        result = (
            np.sqrt(2*self.running_variance[-1]*ln_compl) + (3*self.range_of_rndvar*ln_compl))
        return result

    def update_ct(self):
        # Update ct
        self.current_k += 1
        self.ct.append(self.calc_ct())

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return self.running_mean[-1]

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Should return the array of the variances
    def get_var(self):
        return np.asarray(self.running_variance)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_t


class eba_geo_marg():
    # Initlialize Bernstein with epsilon and delta
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

    # Should add sample to self.samples and should update all the parameters
    def add_sample(self, sample):
        # Update current step
        self.current_t = self.current_t + 1

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

    # Either returns true or false, depending on wheter EBS stopped or not
    # Loop in main program should check this every iteration to determine
    # Loop in application should check for this to be False --> termiante

    def cond_check(self):
        if self.current_k == 0:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    def inner_cond_check(self):

        return self.current_t > np.floor(self.beta**self.current_k)

    # Just a function to calculate c_t for a given time t
    def calc_ct(self):
        return np.sqrt(2*self.running_variance[-1]*self.x/self.current_t)+3*self.range_of_rndvar*self.x/self.current_t

    def update_ct(self):
        # Update ct
        self.current_k += 1
        self.alpha = np.ceil(self.beta**self.current_k) / \
            np.ceil(self.beta**self.current_k-1)
        self.x = -self.alpha*np.log(self.c/3*(self.current_k**self.p))
        self.ct.append(self.calc_ct())

    def get_ct(self):
        return np.asarray(self.ct)

    # Should return the latest estimated mean
    def get_estimate(self):
        return self.running_mean[-1]

    # Should return the array of the estimated means
    def get_mean(self):
        return np.asarray(self.running_mean)

    # Should return the array of the variances
    def get_var(self):
        return np.asarray(self.running_variance)

    # Returns current iteration/ step
    def get_step(self):
        return self.current_t
