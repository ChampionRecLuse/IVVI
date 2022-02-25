import math
import numpy as np
import time
import scipy.stats as spst


def computeDimensions(transition):
    N = len(transition)
    try:
        if transition.ndim == 3:
            M = transition.shape[1]
        else: 
            M = transition[0].shape[0]
    except AttributeError:
        print("Improper matrix shape")
    
    return M, N

def printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))

class MDP(object):

    """
    A Markov Decision Problem

    Let ''M'' = the number of states, ''N'' = the number of actions.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. These can be defined in a variety of
        ways. The simplest is a numpy array that has the shape ``(N, M, M)``,
        though there are other possibilities.
    reward : array
        Reward matrices at different time. Like the transition matrices, these can
        also be defined in a variety of ways. Again the simplest is a numpy
        array that has the shape ``(H, N, M)``, where H = horizon.
    discount : float
        Discount factor. The per time-step discount factor on future rewards.
        Valid values are greater than 0 upto and including 1. If the discount
        factor is 1, then convergence is cannot be assumed and a warning will
        be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a discount factor.
    horizon : int
        The horizon of MDP. It must be an integer.

    Attributes
    ----------
    P : array
        Transition probability matrix.
    R : array
        Reward matrix at a specific time.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    discount : float
        The discount rate on future rewards.
    policy : tuple
        The optimal policy.
    time : float
        The time used to converge to the optimal policy.
    verbose : boolean
        Whether verbose output should be displayed or not.
    """
    
    def __init__(self, transitions, reward, discount, horizon):
        # Initialise a MDP based on the input parameters.

        if discount is not None:
            self.discount = float(discount)
            assert 0 < self.discount <= 1, "Discount rate must be in (0,1]"
            if self.discount == 1:
                print("WARNING: check conditions of convergence. With no disount, convergence can not be assumed.")

        if horizon is not None:
            self.H = int(horizon)
            assert self.H > 0, "The horison of MDP must be positive."

        if reward is not None:
            self.reward = reward

        # compute dimensions
        # M is the number of states, N is the number of actions.
        self.M, self.N = computeDimensions(transitions)
        # compute transition matrix
        self.P = self.computeTransition(transitions)
        # the verbosity is by default turned off
        self.verbose = True
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector 
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

    def computeTransition(self, transition):
        return tuple(transition[a] for a in range(self.N))

    def computeReward(self, reward, t, horizon):
        # compute the reward for the MDP in one state chosing an action at time t.
        # Arguments
        # M = number of states, N = number of actions
        # reward could be an array with 3 dimensions (TxNxM), 
        # each cell means the reward if the system takes a specific action based on 
        # current state and time t, or
        # with 2 dimensions (NxM), which means the reward at all time is same.
        if t > horizon:
            print("The time should be less than or equal to horizon.")
            return
        try:
            if reward.ndim == 2:
                return reward
            if reward.ndim == 3:
                return reward[t-1,:,:]
        except (AttributeError, ValueError):
            print("Check the reward matrix and input time t.")

    def bellmanOperator(self, t, V=None):
        # Apply the Bellman operator on the value function
        # Updates the value and the policy at time t
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If we don't have the information about V, we use the objects V attribute

        if V is None:
            V = self.V
        else:
            # make sure the supplied V has the right shape.
            try:
                assert V.shape in ((self.M,), (1, self.M)), "V is not in right shape."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Get the reward matrix at time t.
        R = self.computeReward(self.reward, t, horizon=self.H)
        # Looping through each action to calculate the Q-value matrix.
        Q = np.empty((self.N, self.M))
        for action in range(self.N):
            # print(V)
            # print(self.P[action])
            # print(np.shape(V))
            # print(np.shape(self.P[action]))
            # print(np.dot(V, self.P[action]))
            Q[action] = R[action] + self.discount * np.dot(V, self.P[action])
        # Get the policy and value
        return (Q.argmax(axis=0), Q.max(axis=0))
    
    def starRun(self):
        if self.verbose:
            printVerbosity('Iteration', 'Variation')

        self.time = time.time()

    def endRun(self):
        # sotre value and policy as tuples
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = time.time() - self.time

    def run(self):
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True


class FiniteHorizon(MDP):

    """A MDP solved using the finite-horizon backwards induction algorithm.

    Parameters
    ----------
    transitions : array
        Transition probability matrix. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    H : int
        Number of periods. Must be greater than 0.
    h : array, optional
        Terminal reward. Default: a vector of zeros.

    Data Attributes
    ---------------
    V : array
        Optimal value function. Shape = (H+1, M). ``V[n, :]`` = optimal value
        function at stage ``n`` with stage in {0, 1...H-1}. ``V[H, :]`` value
        function for terminal stage.
    policy : array
        Optimal policy. ``policy[n, :]`` = optimal policy at stage ``n`` with
        stage in {0, 1...N}. ``policy[H, :]`` = policy for stage ``H``.
    time : float
        used CPU time

    Notes
    -----
    In verbose mode, displays the current stage and policy transpose.

    """

    def __init__(self, transitions, reward, discount, H, h=None):
        # Initialize a finite horizon MDP.
        MDP.__init__(self, transitions, reward, discount, H)
        # remove the iteration counter, it is not meaningful for backwards induction
        del self.iter
        # We have value vectors for each time step up to the horizon
        # Each row for V is the value vector for a specific time
        self.V = np.zeros((self.H + 1, self.M))
        # We have policy vectors for each time step before the horizon,
        # when we reach the horizon we don't need to make decisions
        # Each row for policy is the policy vector for a specific time
        self.policy = np.empty((self.H, self.M), dtype=int)
        # Set the reward for the final transition
        if h is not None:
            self.V[self.H, :] = h

    def run(self):
        # Run the finite horizon backwards algorithm
        self.time = time.time()
        # loop through each time period
        for t in range(self.H, 0, -1):
            W, X = self.bellmanOperator(t, self.V[t, :])
            stage = t - 1
            self.V[stage, :] = X
            self.policy[stage, :] = W
            if self.verbose:
                print(("stage: %s, policy: %s") % (stage, self.policy[stage, :].tolist()))
            
            # update time spent running
            self.time = time.time() - self.time




def phi(x,a):

    """
    The feature map
    """

    return

def psi(z):

    """
    The feature map
    """

    return

def reward(H,S,A):

    """
    The reward function at different time

    Parameters
    ----------
    H : int
        The horizon of MDP.
    S : array
        S is the finite discrete state space. It's a numpy matrix with shape ''(d_{x}, M)'', where every column is a state.
    A : array
        A is the finite discrete action space. It's a numpy matrix with shape ''(d_{a}, N)'', wherer every column is an action.

    Output
    ----------
    reward : array
        A reward matrix with shape ''(H, N, M)'',
        where ''M'' = the number of states, ''N'' = the number of actions.
    """
    return 


def compute_W_sad(phi, psi, eta_phi, eta_psi, x, a, z, xx, max_iter, W_0, K_0):

    """ Compute W^{sad} straightly

    Parameters
    ----------
    phi : function
        Feature map. phi : S * A -> R^{d_{phi}}
    psi : function
        Feature map. psi : Z -> R^{d_{psi}}
    eta_phi: array
        Stepsize in the stochastic gradient. It is a numpy array with shape ''(1, T)''
    eta_psi: array
        Stepsize in the stochastic gradient. It is a numpy array with shape ''(1, T)''
    x : array
        It's a given state sample matrix. It can be a numpy matirx with shape ''(d_{x}, T)'', where T is the parameter iter.
    a : array
        It's a given action sample matrix. It can be a numpy matirx with shape ''(d_{a}, T)'', where T is the parameter iter.
    z : array
        It's a given instrumental variable sample matrix. It can be a numpy matirx with shape ''(d_{z}, T)'', where T is the parameter iter.
    xx : array
        It's a given next state sample matrix. It can be a numpy matirx with shape ''(d_{x}, T)'', where T is the parameter iter.
    max_iter : int
        Number of iterations.
    W_0 : array
        It's an initial guess for output W. W_0 can be a numpy matirx with shape ''(d_{x}, d_{phi})''.
    K_0 : array
        It's an initial guess for K. K_0 can be a numpy matirx with shape ''(d_{x}, d_{psi})''.

    Output
    --------
    W_now : array
        It's an estimation matrix for the true W_sad with shape ''(d_{x}, d_{phi})''
    """

    W_now, W_next = np.matrix(W_0), np.matrix(W_0)
    K_now, K_next = np.matrix(K_0), np.matrix(K_0)
    for t in range(max_iter):
        Phi = np.matrix(phi(x[:, t], a[:, t]))
        Psi = np.matrix(psi(z[:, t]))
        # W_next = W_now - eta_phi[t] * (K_now.dot(Psi * np.transpose(Phi)))
        W_next = W_now - eta_phi[t] * (K_now.dot(np.dot(Psi, np.transpose(Phi))))
        K_next = K_now + eta_psi[t] * (K_now.dot((np.dot(Psi, np.transpose(Psi)))) + np.dot(xx[:, t], np.transpose(Psi)) - W_now.dot((np.dot(Phi, np.transpose(Psi)))))
        W_now = W_next.copy()
        K_now = K_next.copy()
        # W_now = W_next
        # K_now = K_next
    
    return W_now


def transition_matrix(phi, W, S, A, sigma):

    """
    In this case, we assume state space S and action space A are both finite discrete. 
    We use W * phi to estimate function F. Our trasition matrix is from $P_{F}(\dot|x_{h}) = N(F(x_{h},a_{h}),\sigma^{2}I_{d_{x}})$.
    In order to compute quickly, we just use the pdf of multivariate normal distribution to approximate the trasition probability for every state.

    Parameters
    ----------
    phi : function
        Feature map. phi : S * A -> R^{d_{phi}}
    W : array
        W is a matrix from Phase I. We use W * phi to estimate function F.
    S : array
        S is the finite discrete state space. It's a numpy matrix with shape ''(d_{x}, M)'', where every column is a state.
    A : array
        A is the finite discrete action space. It's a numpy matrix with shape ''(d_{a}, N)'', where every column is an action.
    sigma : float 
        sigma^{2}I is the covariance for Normal Distribution.

    Output
    ----------
    T : array
        T is the transition matrix from (action, state) to state. It's a numpy matrix with shape ''(N, M, M)''.
    """
    d_x, M = np.shape(S)
    d_a, N = np.shape(A)
    T = np.zeros((N, M, M))
    covariance = (sigma ** 2) * np.eye(d_x)
    for i in range(N):
        for j in range(M):
            P = sum(spst.multivariate_normal.pdf(S[:,k], mean = np.dot(W, phi(S[:, j], A[:, i])), cov = covariance) for k in range(M))
            for k in range(M):
                T[i,j,k] = spst.multivariate_normal.pdf(S[:,k], mean = np.dot(W, phi(S[:, j], A[:, i])), cov = covariance) / P

    return T

def transition_matrix2(phi, W, S, A):

    """
    In this case, we assume state space S and action space A are both finite discrete.
    We use W * phi to estimate function F. Our trasition matrix is from $P_{F}(\dot|x_{h}) = F(x_{h},a_{h}) + (2 * Bernoulli(p=1/2) - 1)$.
    In order to compute quickly, we just use the pdf of multivariate normal distribution to approximate the trasition probability for every state.

    Parameters
    ----------
    phi : function
        Feature map. phi : S * A -> R^{d_{phi}}
    W : array
        W is a matrix from Phase I. We use W * phi to estimate function F.
    S : array
        S is the finite discrete state space. It's a numpy matrix with shape ''(d_{x}, M)'', where every column is a state.
    A : array
        A is the finite discrete action space. It's a numpy matrix with shape ''(d_{a}, N)'', where every column is an action.

    Output
    ----------
    T : array
        T is the transition matrix from (action, state) to state. It's a numpy matrix with shape ''(N, M, M)''.
    """
    d_x, M = np.shape(S)
    d_a, N = np.shape(A)
    T = np.zeros((N, M, M))
    bias = int(0 - S[:, 0])

    for i in range(N):
        for j in range(M):
            center = np.floor(np.dot(W, phi(S[:, j], A[:, i])))
            if center + 1 > S[:, M-1] and center - 1 >= S[:, M-1]:
                T[i, j, M -1] = 1
            elif center + 1 >= S[:, M-1] and center - 1 < S[:, M-1]:
                T[i, j, M - 1] = 0.5
                T[i, j, int(center) - 1 + bias] = 0.5
            elif center - 1 <= S[:, 0] and center + 1 > S[:, 0]:
                T[i, j, 0] = 0.5
                T[i, j, int(center) + 1 + bias] = 0.5
            elif center + 1 <= S[:, 0]:
                T[i, j, 0] = 1
            else:
                T[i, j, int(center) - 1 + bias] = 0.5
                T[i, j, int(center) + 1 + bias] = 0.5
    for i in range(N):
        T[i] = T[i].T
    return T


# def main():
    # State space
    # S = {}

    # # Action space
    # A = {}

    # # example
    # x = np.matrix([1 / i**3 for i in range(1,101)])
    # a = np.matrix([1 / i ** 3 for i in range(1,101)])
    # xx = np.c_[x[:, 1:], 0.005]
    # z = np.matrix([1 for i in range(100)])

    # # stepsize
    # eta_psi = np.array([1/ i**2 for i in range(1,101)])
    # eta_phi = np.array([1/ i**2 for i in range(1,101)])

    # # iterations
    # max_iter = 100

    # # initial estimates
    # W_0 = np.matrix([20])
    # K_0 = np.matrix([30])

    # # reward matrix
    # R =

    # W_sad = compute_W_sad(phi, psi, eta_phi, eta_psi, x, a, z, xx, max_iter, W_0, K_0)
    # print(W_sad)
    # for i in range(100):
    #     for j in range(100):
    #         print(W_sad * phi(x[:, i], a[:, j]))


# if __name__ == '__main__':
#     main()


