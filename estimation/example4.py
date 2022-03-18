import math
import numpy as np
import time
import scipy.stats as spst
import matplotlib.pyplot as plt

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

    W_now = W_0.copy()
    K_now = K_0.copy()
    W_next = W_0.copy()
    K_next = K_0.copy()

    for t in range(max_iter):
        Phi = np.matrix(phi(x[:, t],a[:, t]))
        Psi = np.matrix(psi(x[:,t],z[:, t]))
        W_next = W_now - eta_phi[t] * np.dot(K_now,np.dot(Psi, np.transpose(Phi)))
        K_next = K_now + eta_psi[t] * (np.dot(K_now,np.dot(Psi, np.transpose(Psi))) + np.dot([xx[:, t]], np.transpose(Psi)) - np.dot(W_now, np.dot(Phi, np.transpose(Psi))))
        W_now = W_next.copy()
        K_now = K_next.copy()

    return W_now


# Use mini batch stochastic gradient descent.
# One thing is that transition function x+2a does not work, but 0.95x+2a works.
def compute_W_sad2(phi, psi, eta_phi, eta_psi, x, a, z, xx, max_iter, W_0, K_0, size):

    """ Compute W^{sad} straightly

    Parameters
    ----------
    phi : function
        Feature map. phi : S * A -> R^{d_{phi}}
    eta_phi: array
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

    Output
    --------
    W_now : array
        It's an estimation matrix for the true W_sad with shape ''(d_{x}, d_{phi})''
    """

    W_now = W_0.copy()
    K_now = K_0.copy()
    W_next = W_0.copy()
    K_next = K_0.copy()

    for i in range(int(max_iter/size)):
        mean_x = np.mean(x[:,i * size: (i + 1) * size])
        mean_a = np.mean(a[:,i * size: (i + 1) * size])
        mean_xx = np.mean(xx[:,i * size: (i + 1) * size])
        mean_z = 0
        mean_x2 = np.mean(x[:, i * size:(i + 1) * size] * x[:, i * size:(i + 1) * size])
        mean_xa = np.mean(x[:, i * size:(i + 1) * size] * a[:, i * size:(i + 1) * size])
        mean_zx = np.mean(z[:, i * size:(i + 1) * size] * x[:, i * size:(i + 1) * size])
        mean_z2 = np.mean(z[:, i * size:(i + 1) * size] * z[:, i * size:(i + 1) * size])
        mean_za = np.mean(z[:, i * size:(i + 1) * size] * a[:, i * size:(i + 1) * size])
        mean_xxz = np.mean(xx[:, i * size:(i + 1) * size] * z[:, i * size:(i + 1) * size])
        mean_xxx = np.mean(xx[:, i * size:(i + 1) * size] * x[:, i * size:(i + 1) * size])
        A = np.array([[1, mean_x, mean_a],[mean_x, mean_x2, mean_xa], [mean_z, mean_zx, mean_za]])  # matrix A
        B = np.array([[1,mean_x, mean_z],[mean_x, mean_x2, mean_zx], [mean_z, mean_zx, mean_z2]])  # matrix B
        C = np.array([[mean_xx, mean_xxx, mean_xxz]])  # matrix C

        for j in range(100):
            K_next = K_now +  eta_psi[j] * (np.dot(K_now, B) + C - np.dot(W_now, np.transpose(A)))
            K_now = K_next.copy()
        W_next = W_now - eta_phi[i] * np.dot(K_now, A)
        W_now = W_next.copy()

    return W_now


def phi(x,a):

    """
    The feature map
    :param x: a state. The shape is ''(d_{x}, 1)''
    :param a: an action. The shape is ''(d_{a}, 1)''
    :return:
    """
    return np.array([[1], x, a])

def psi(x,z):

    """
    The feature map
    :param z: an instrumental variable. The shape is ''(d_{z}, 1)''
    :return:
    """
    return np.array([[1], x, z])

def generate_action(x,z,e, beta, gamma, delta):

    """
    A function to generate action while collecting data.
    """

    return np.clip(np.random.normal(z+e, 1), -1, 1)

def F(x, a):

    """
    A deterministic transition function we want to approach.
    """

    return 0.9 * x + 2 * a
    # return np.log(np.abs(x+2*a))


# Data collection
max_iter = 40000
size = 256
beta = -0.1
gamma = 0.1
delta = 0.1

x = np.array([[0]])  # initial state
z = np.array([[np.random.normal()]])
e = np.array([[np.random.normal()]])
a = np.array(generate_action(x,z,e,beta,gamma,delta))  # get the action
x_next = np.array(F(x,a)+e)  # get the next state
xx = x_next.copy()  # store the next state

# Collecting the data step by step
for i in range(max_iter):
    x = np.hstack((x,x_next))
    z_next = np.array([[np.random.normal()]])
    e_next = np.array([[np.random.normal()]])
    z = np.hstack((z,z_next))
    e = np.hstack((e,e_next))
    a_next = np.array([generate_action(x[:,i+1],z[:,i+1],e[:,i+1],beta,gamma,delta)])
    x_next = np.array(F(x_next,a_next)+e_next)
    a = np.hstack((a,a_next))
    xx = np.hstack((xx,x_next))

# print(np.shape(e))
# print(np.shape(x))
# print(np.shape(z))
# print(np.shape(a))
# print(np.shape(xx))

# Compute Closed Form
# The closed form for W^{*} is CB^{-1}A(A^{T}B^{-1}A)^{-1}
mean_x = np.mean(x)
mean_xx = np.mean(xx)
mean_a = np.mean(a)
mean_z = 0
mean_x2 = np.mean(x*x)
mean_xa = np.mean(x*a)
mean_z2 = np.mean(z*z)
mean_zx = np.mean(z*x)
mean_za = np.mean(z*a)
mean_xxz = np.mean(xx*z)
mean_xxx = np.mean(xx*x)

# use sample to estimate covariance matrices
A = np.array([[1, mean_x, mean_a],[mean_x, mean_x2, mean_xa], [mean_z, mean_zx, mean_za]])  # matrix A
B = np.array([[1,mean_x, mean_z],[mean_x, mean_x2, mean_zx], [mean_z, mean_zx, mean_z2]])  # matrix B
C = np.array([[mean_xx, mean_xxx, mean_xxz]])  # matrix C
W_front = np.dot(np.dot(C,np.linalg.inv(B)),A)  # compute CB^{-1}A
W_back = np.dot(np.dot(np.transpose(A),np.linalg.inv(B)), A)  # compute A^{T}B^{-1}A
W_star = np.dot(W_front, np.linalg.inv(W_back))
print(W_star)
K_star = np.dot(np.dot(W_star,np.transpose(A))-C,np.linalg.inv(B))
print(K_star)



# Compute W by using Gradient Descent
W_0 = np.array([[1,1,1]])  # initial guess
K_0 = np.array([[0,0,0]])  # initial guess
eta_phi = np.zeros(max_iter)
eta_psi = np.zeros(max_iter)
# setup for stepsize
for i in range(max_iter):
    eta_psi[i] = -1/(1+i)
    eta_phi[i] = 0.05


W_sad = compute_W_sad2(phi, psi, eta_phi, eta_psi, x, a, z, xx, max_iter, W_0, K_0, size)
print('--------------')
print(W_sad)

