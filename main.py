from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution, mp_plot

X = RandomMatrix(2000, 1000)
X.covariance()

x, rho = mp_distribution(X.q, X.lambda_p, X.lambda_m)
mp_plot(x, rho, X.lambda_m, X.lambda_p, X.nonzero_eig)