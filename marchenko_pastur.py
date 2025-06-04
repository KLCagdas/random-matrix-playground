import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from scipy.signal import unit_impulse

'''
parameters
'''
# number of states
N = 1000
# number of observations
T = 2000
# ratio of num. of states to num. of observations
q = N/T

# non-symmetric random matrix
X = np.random.normal(0, 1, (T, N))
# NxN covariance
Y = 1/T * X @ X.T
# compute the eigenvalues
eigval = np.linalg.eigvalsh(Y)

# marcenko-pastur bounds
lambda_p = (1 + sqrt(q))**2
lambda_m = (1 - sqrt(q))**2

def mp_distribution(x, q, lambda_p, lambda_m):
    # you can also change below to true false statement
    rho = np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
    if q > 1:
        #rho[np.where(x == 0)] += (q - 1) / q
        plt.axvline(x=0, linestyle='-', color='sienna', label='_nolegend_')
    return rho

x_val = np.linspace(lambda_m, lambda_p + 1, 1000)
rho_x = mp_distribution(x_val, q, lambda_p, lambda_m)

# number of zero eigenvalues
# this is not necessary for T > N 
nonzero_eig = eigval[eigval > 1e-10]
num_zero_eig = np.array(np.where(eigval < 1e-10)).size
print(num_zero_eig)

# marchenko-pastur plot
plt.plot(x_val, rho_x, 'sienna', label='Theoretical MP Curve')
# bins for eigenvalue distribution
bins = np.linspace(lambda_m, lambda_p, num=50)
# histogram of eigenvalues
plt.hist(nonzero_eig, bins=bins, density=True, color='bisque', label='Eigenvalue Distribution')
# plot specifications
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho_{MP}(x)$')
plt.title('Marchenko-Pastur (MP) Distribution')

plt.axvline(x=lambda_m, linestyle='--', color='k', label='Theoretical Bound', alpha=0.5, linewidth=0.8)
plt.axvline(x=lambda_p, linestyle='--', color='k', label='_nolegend_', alpha=0.5, linewidth=0.8)

plt.legend()

print(nonzero_eig)

plt.show()

'''
FOR N>T NORMALIZE EIGVALS 
average over many random matrices
look at arabind'or philipp's papers
'''