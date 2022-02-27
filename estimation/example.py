"""
Example for the Instrumental Variable Value Iteration for Causal Offline Reinforcement Learning
"""

import math
import numpy as np
import time
import scipy.stats as spst
import mdp

def F(x, a):

    """
    A deterministic transition function we want to approach.
    """
    return 0.3 * x[0] + 0.2 * a[0]
    # return np.log(np.abs(x[0] + 2 * a[0]))

def generate_action(x, z, e, beta, gamma, delta):

    """
    A function to generate action while collecting data.
    """
    # U = np.random.rand()
    # P_0 = np.exp(beta * x + gamma * z + delta * e) / (1 + np.exp(beta * x + gamma * z + delta * e))
    # if U <= P_0:
    #     return 0
    # else:
    #     return 1
    # return np.clip(np.random.normal(0,beta * float(x)**2+gamma * float(z)**2+delta * float(e)**2),-1.,1.)
    return np.random.normal(-beta * x + gamma * z + delta * e, 1)
    # return np.tanh(0.1 * (-x+z+e))


def phi(x, a):

    """
    The feature map
    :param x: a state. The shape is ''(d_{x}, 1)''
    :param a: an action. The shape is ''(d_{a}, 1)''
    :return:
    """
    return np.array([[1], x, a])

def psi(z):

    """
    The feature map
    :param z: an instrumental variable. The shape is ''(d_{z}, 1)''
    :return:
    """
    return np.array([[1], z])

def reward(H, S, A):
    """
    The reward function at different time
    :param H: The horizon of MDP.
    :param S: S is the finite discrete state space. It's a numpy matrix with shape ''(d_{x}, M)'', where every column is a state.
    :param A: A is the finite discrete action space. It's a numpy matrix with shape ''(d_{a}, N)'', wherer every column is an action.
    :return: A reward matrix with shape ''(H, N, M)'', where ''M'' = the number of states, ''N'' = the number of actions.
    """
    d_x, M = np.shape(S)
    d_a, N = np.shape(A)
    R = np.zeros((H, N, M))
    for i in range(N):
        for j in range(M):
            R[H - 1, i, j] = S[:, j]
            # R[H - 1, i, j] = max(S[:, j] + A[:, i], 0)

    return R

# Data Collection
horizon = 10000
max_iter = 1000
beta = 0.1
gamma = 1
delta = 1

x = np.array([[0]])  # The initial state
z = np.array([[np.random.normal()]])
e = np.array([np.random.normal()])
# a = np.array([[generate_action(x, z, e, beta, gamma, delta)]])
a = np.array(generate_action(x, z, e, beta, gamma, delta))
x_next = np.array([[F(x[:, 0], a[:, 0]) + e[0]]])
xx = np.array([[F(x[:, 0], a[:, 0]) + e[0]]])


for i in range(max_iter):
    x = np.hstack((x, x_next))
    z_next = np.array([[np.random.normal()]])
    z = np.hstack((z, z_next))
    e_next = np.array([np.random.normal()])
    e = np.hstack((e, e_next))
    # a_next = np.array([[generate_action(x[:, i + 1], z[:, i + 1], e[i + 1], beta, gamma, delta)]])
    a_next = np.array([generate_action(x[:, i + 1], z[:, i + 1], e[i + 1], beta, gamma, delta)])
    a = np.hstack((a, a_next))
    x_next = np.array([[F(x[:, i+1], a[:, i+1]) + e[i+1]]])
    # x_next = np.array([x[:, i + 1]+ 2 * a[:, i + 1] + e[i + 1]])
    xx = np.hstack((xx, x_next))

# print(x)
# print(a)
# print(e)
# print(z)
# print(np.shape(xx[:,1]))
eta_phi = np.zeros(max_iter)
eta_psi = np.zeros(max_iter)
for i in range(max_iter):
    eta_phi[i] = 1/(1+i)
    eta_psi[i] = eta_phi[i]
W_0 = np.array([[1, 1, 1]])
K_0 = np.array([[1, 1]])

W_sad= mdp.compute_W_sad(phi, psi, eta_phi, eta_psi, x, a, z, xx, max_iter, W_0, K_0)
print(W_sad)


mean_x = np.mean(x)
mean_xx = np.mean(xx)
mean_a = np.mean(a)
mean_z = 0
mean_zz = np.mean(np.power(z,2))
mean_zx = np.mean(z*x)
mean_za = np.mean(z*a)
mean_xxz = np.mean(xx*z)
print(mean_x)
print(mean_xx)
print(mean_a)
print(mean_z)
A = np.array([[1, mean_x, mean_a],[mean_z, mean_zx, mean_za]])
B = np.array([[1,mean_z],[mean_z,mean_zz]])
C = np.array([[mean_xx, mean_xxz]])
W_front = np.dot(np.dot(C,np.linalg.inv(B)),A)
W_back = np.dot(np.dot(np.transpose(A),np.linalg.inv(B)), A)
W_star = np.dot(W_front, np.linalg.inv(W_back))
print(W_star)
mu_IV = np.min(np.linalg.eig(W_back)[0])
print(mu_IV)