import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt

def mp_distribution(q, q_p=None, gamma=None):
    """
    Computes the theoretical Marchenko-Pastur (MP) distribution.

    Parameters:
        q (float): Ratio parameter.
        q_p (float, optional): Ratio parameter when projecting onto m directions.
        gamma (float, optional): Scaling factor when projecting onto m directions.

    Returns:
        x (ndarray): x values for MP distribution.
        rho (ndarray): MP density values.
        lambda_m (float): Lower bound of MP support.
        lambda_p (float): Upper bound of MP support.
    """
    if q_p is not None and gamma is not None:
        # Modified MP distribution
        lambda_p = (1 + sqrt(q_p))**2
        lambda_m = (1 - sqrt(q_p))**2
        x = np.linspace(lambda_m, lambda_p + 1, 1000)
        rho = 1 / gamma * np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
    else:
        # Standard MP distribution
        lambda_p = (1 + sqrt(q))**2
        lambda_m = (1 - sqrt(q))**2
        x = np.linspace(lambda_m, lambda_p + 1, 1000)
        rho = np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
        if q > 1:
            rho *= q
    return x, rho, lambda_m, lambda_p

def mp_plot(x, rho, lambda_m, lambda_p, eigvals):
    # plot the marchenko-pastur distribution
    plt.plot(x, rho, 'sienna', label='Semi-analytic Distribution')
    # bins for eigenvalue distribution
    bins = np.linspace(lambda_m, lambda_p, num=40)
    # plot the histogram of eigenvalues
    counts, _, _ = plt.hist(eigvals, bins=bins, density=True, color='bisque', label='Empirical Distribution')
    # label the plot
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho_{MP}(x)$')
    plt.title('Marchenko-Pastur (MP) Distribution')
    # plot the theoretical MP bounds
    plt.axvline(x=lambda_m, linestyle='--', color='k', label='Analytic Bounds', alpha=0.5, linewidth=0.8)
    plt.axvline(x=lambda_p, linestyle='--', color='k', label='_nolegend_', alpha=0.5, linewidth=0.8)
    # fix axes limits
    plt.xlim(lambda_m - 0.1, lambda_p + 0.1)
    plt.ylim(0, max(np.max(rho), np.max(counts)) * 1.1)
    plt.legend()
    plt.show()
 
