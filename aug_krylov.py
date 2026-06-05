# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:30:49 2024

@author: Francisco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import gmres

def generate_diagonal_matrix(n):
    diagonal_entries = np.zeros(n)
    for i in range(1, n+1):
        if i > 4:
            diagonal_entries[i-1] = i / n
        else:
            diagonal_entries[i-1] = 0.05 * (i / n)
    return diags(diagonal_entries, 0, format='csr')

def generate_random_vector(n):
    return np.random.rand(n)


def plot_convergence(iterations, residual_norms, title):
    plt.figure()
    plt.semilogy(iterations, residual_norms, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title(title)
    plt.grid()
    plt.show()
    
    
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.r_k = []
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            #print('iter %3i\trk = %s' % (self.niter, str(rk)))
            self.r_k.append(rk)


# Parameters
n = 200
A = generate_diagonal_matrix(n)
b = generate_random_vector(n)
x0 = np.zeros(n)

# Solve using standard GMRES
counter1 = gmres_counter()
x_std, info_std = gmres(A, b, x0=x0, maxiter=None, callback=counter1)

k_1 = counter1.niter 
iterations_std = np.arange(k_1)
r_k_1 = counter1.r_k / (np.linalg.norm(b))

print(k_1)
print(len(r_k_1))


# Solve using GMRES with restarts
counter2 = gmres_counter()
restart = 40
x_restart, info_restart = gmres(A, b, x0=x0, maxiter=None, restart=restart, callback=counter2)

k_2 = counter2.niter
r_k_2 = counter2.r_k / (np.linalg.norm(b))


# Plot convergence

plt.figure()
plt.semilogy(np.arange(k_1), r_k_1, '-b', label='GMRES(40)')
plt.semilogy(np.arange(k_2), r_k_2, '-r', label='GMRES(inf)')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title("Convergence comparison")
plt.grid()
plt.legend()
plt.show()














