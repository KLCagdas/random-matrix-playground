import numpy as np
from math import pi
import matplotlib.pyplot as plt

# theoretical Marcenko-Pastur (MP) distribution
def mp_distribution(x, q, lambda_p, lambda_m):
    # you can also change below to true false statement
    rho = np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
    if q > 1:
        # rescale the density since non-zero eigenvalues are removed
        rho *= q
    return rho

def mp_plot(x, rho, lambda_m, lambda_p, eigvals):
    # plot the marchenko-pastur distribution
    plt.plot(x, rho, 'sienna', label='Theoretical MP Curve')
    # bins for eigenvalue distribution
    bins = np.linspace(lambda_m, lambda_p, num=40)
    # plot the histogram of eigenvalues
    plt.hist(eigvals, bins=bins, density=True, color='bisque', label='Eigenvalue Distribution')
    # label the plot
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho_{MP}(x)$')
    plt.title('Marchenko-Pastur (MP) Distribution')
    # plot the theoretical MP bounds
    plt.axvline(x=lambda_m, linestyle='--', color='k', label='Theoretical Bound', alpha=0.5, linewidth=0.8)
    plt.axvline(x=lambda_p, linestyle='--', color='k', label='_nolegend_', alpha=0.5, linewidth=0.8)

    plt.legend()
    plt.show()