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
Y = 1/T * X.T @ X
# compute the eigenvalues
eigval = np.linalg.eigvalsh(Y)
print(eigval)

# marcenko-pastur bounds
lambda_p = (1 + sqrt(q))**2
lambda_m = (1 - sqrt(q))**2

def mp_distribution(x, q, lambda_p, lambda_m):
    # you can also change below to true false statement
    rho = np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
    if q > 1:
        #rho[np.where(x == 0)] += (q - 1) / q
        plt.axvline(x=0, linestyle='-', color='chocolate')
    return rho

x_val = np.linspace(0, lambda_p + 1, 1000)
rho_x = mp_distribution(x_val, q, lambda_p, lambda_m)

print(np.array(np.where(eigval < 1e-10)).size)


plt.plot(x_val, rho_x, 'chocolate')
bins = np.linspace(lambda_m, lambda_p, num=20)
plt.hist(eigval, bins=bins, density=True, color='bisque')
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho_{MP}(x)$')
plt.title('Marchenko-Pastur (MP) Distribution')
plt.legend(['Theoretical MP Curve', 'Eigenvalue Distribution'])
plt.show()

'''
FOR N>T NORMALIZE EIGVALS 
get rid of zero eigenvalues (N-T) etc.
average over many random matrices
look at arabind'or philipp's papers
it may make sense to bin the eigvals into order of magnitude
'''

'''
QUESTIONS
1. why do we need to get rid of zero eigvals?
2. why is there no eigval density for large values, i.e., after 3?
'''