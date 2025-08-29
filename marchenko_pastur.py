import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt

# theoretical Marcenko-Pastur (MP) distribution
def mp_distribution_half(q):
    lambda_p = (1 + sqrt(q))**2
    lambda_m = (1 - sqrt(q))**2
    # x values for MP distribution
    x = np.linspace(lambda_m, lambda_p + 1, 1000)
    # the line below can be replaced by a true-false statement
    rho = np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)
    if q > 1:
        # rescale the density since non-zero eigenvalues are removed
        rho *= q
    return x, rho, lambda_m, lambda_p
    
# theoretical Marcenko-Pastur (MP) distribution
def mp_distribution_m(q, q_p, gamma):
    lambda_p = (1 + sqrt(q_p))**2
    lambda_m = (1 - sqrt(q_p))**2
    # x values for MP distribution
    x = np.linspace(lambda_m, lambda_p + 1, 1000)
    # the line below can be replaced by a true-false statement
    rho = 1 / gamma * np.sqrt(np.maximum((lambda_p - x)*(x - lambda_m), 0)) / (2 * pi * q * x)

    return x, rho, lambda_m, lambda_p

def mp_plot(x, rho, lambda_m, lambda_p, eigvals):
    # plot the marchenko-pastur distribution
    plt.plot(x, rho, 'sienna', label='Semi-analytic Distribution')
    # bins for eigenvalue distribution
    bins = np.linspace(lambda_m, lambda_p, num=40)
    # plot the histogram of eigenvalues
    counts, edges, _ = plt.hist(eigvals, bins=bins, density=True, color='bisque', label='Empirical Distribution')
    # calculate if histogram sums up to 1
    bin_widths = np.diff(edges)
    hist_sum = np.sum(counts * bin_widths)
    print(f"Histogram sums up to: {hist_sum:.4f}")
    # label the plot
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho_{MP}(x)$')
    plt.title('Marchenko-Pastur (MP) Distribution')
    # plot the theoretical MP bounds
    plt.axvline(x=lambda_m, linestyle='--', color='k', label='Analytic Bounds', alpha=0.5, linewidth=0.8)
    plt.axvline(x=lambda_p, linestyle='--', color='k', label='_nolegend_', alpha=0.5, linewidth=0.8)

    plt.legend()
    plt.show()
 
