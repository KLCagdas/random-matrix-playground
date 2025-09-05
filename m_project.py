
from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution
import numpy as np
import matplotlib.pyplot as plt

# create a random matrix
X = RandomMatrix(1000, 2000)
# compute the eigvals of its covariance
X.covariance()
# calculate the Marcenko-Pastur distribution of the Wishart matrix (before projection)
x, rho, lambda_low, lambda_upp = mp_distribution(X.q)


def plot_mp_distributions(x, rho, lambda_low, lambda_upp, x_m, rho_m, lambda_low_m, lambda_upper_m, nonzero_eig, nonzero_eig_m, m):
    # Define colors for MP plots and histograms
    mp_color_before = '#A0522D'      # soft sienna
    mp_color_after = '#4169E1'       # soft royal blue
    hist_color_before = '#FFDAB9'    # peach puff (soft orange)
    hist_color_after = '#B0C4DE'     # light steel blue (soft blue)

    # plot both MP distributions and the eigenvalue histograms on the same figure
    plt.figure(figsize=(8, 5))
    plt.plot(x, rho, color=mp_color_before, label='MP Before Projection')
    plt.plot(x_m, rho_m, color=mp_color_after, label=f'MP After Projection (m={m})')
    # bins for eigenvalue distributions
    bins_before = np.linspace(lambda_low, lambda_upp, num=40)
    bins_after = np.linspace(lambda_low_m, lambda_upper_m, num=40)
    plt.hist(nonzero_eig, bins=bins_before, density=True, color=hist_color_before, label='Empirical (Original)', alpha=0.6)
    plt.hist(nonzero_eig_m, bins=bins_after, density=True, color=hist_color_after, label=f'Empirical (Projected, m={m})', alpha=0.6)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho_{MP}(x)$')
    plt.title(f'MP Distribution Before and After Projection (m={m})')
    plt.axvline(x=lambda_low, linestyle='--', color='k', label='MP Bounds Before', alpha=0.5, linewidth=0.8)
    plt.axvline(x=lambda_upp, linestyle='--', color='k', label='_nolegend_', alpha=0.5, linewidth=0.8)
    plt.axvline(x=lambda_low_m, linestyle='--', color='gray', label='MP Bounds After', alpha=0.5, linewidth=0.8)
    plt.axvline(x=lambda_upper_m, linestyle='--', color='gray', label='_nolegend_', alpha=0.5, linewidth=0.8)
    plt.xlim(min(lambda_low, lambda_low_m) - 0.1, max(lambda_upp, lambda_upper_m) + 0.1)
    plt.ylim(0, max(np.max(rho), np.max(rho_m)) * 1.1)
    plt.legend()
    plt.show()

# List of m values to try
m_values = [100, 300, 500, 700, 900]
for m in m_values:
    gamma, nonzero_eig_m = X.m_projector(0.4, m)
    x_m, rho_m, lambda_low_m, lambda_upper_m = mp_distribution(X.q, X.q_p, gamma)
    plot_mp_distributions(
        x, rho, lambda_low, lambda_upp,
        x_m, rho_m, lambda_low_m, lambda_upper_m,
        X.nonzero_eig, nonzero_eig_m,
        m
    )