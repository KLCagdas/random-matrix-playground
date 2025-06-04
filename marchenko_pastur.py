import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from scipy.signal import unit_impulse

'''
parameters
'''
# number of variables
N = 6000
# number of observations
T = 3000
# ratio of num. of states to num. of observations
q = N/T

# non-symmetric random matrix
X = np.random.normal(0, 1, (T, N))
# NxN covariance
Y = 1/T * X.T @ X
# compute the eigenvalues
eigval = np.linalg.eigvalsh(Y)

# Marcenko-Pastur bounds
lambda_p = (1 + sqrt(q))**2
lambda_m = (1 - sqrt(q))**2

# theoretical Marcenko-Pastur (MP) distribution
def mp_distribution(x, q, lambda_p, lambda_m):
    # you can also change below to true false statement
    rho = np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
    if q > 1:
        rho *= q
    return rho

# x values for MP distribution
x = np.linspace(lambda_m, lambda_p + 1, 1000)
# expected MP distribution of the eigenvalues
rho = mp_distribution(x, q, lambda_p, lambda_m)


# number of zero eigenvalues
nonzero_eig = eigval[eigval > 1e-10]
# check the number of zero eigenvalues
num_zero_eig = np.array(np.where(eigval < 1e-10)).size

# marchenko-pastur plot
plt.plot(x, rho, 'sienna', label='Theoretical MP Curve')
# bins for eigenvalue distribution
bins = np.linspace(lambda_m, lambda_p, num=60)
# histogram of eigenvalues
plt.hist(nonzero_eig, bins=bins, density=True, color='bisque', label='Eigenvalue Distribution')
# plot specifications
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho_{MP}(x)$')
plt.title('Marchenko-Pastur (MP) Distribution')
# theoretical MP bounds
plt.axvline(x=lambda_m, linestyle='--', color='k', label='Theoretical Bound', alpha=0.5, linewidth=0.8)
plt.axvline(x=lambda_p, linestyle='--', color='k', label='_nolegend_', alpha=0.5, linewidth=0.8)

plt.legend()
plt.show()
