# %% Imports
from zuko.transforms import BernsteinTransform
import torch
from matplotlib import pyplot as plt

# %% Globals
M = 10
batch_size = 10
theta = torch.rand(size=(M,)) * 500  # creates a random parameter vector

# %% Test
bpoly = BernsteinTransform(theta=theta, linear=False)

x = torch.linspace(-11, 11, 2000)
y = bpoly(x)

adj = bpoly.log_abs_det_jacobian(x, y).detach()
J = torch.diag(torch.autograd.functional.jacobian(bpoly, x)).abs().log()

# %% Plot

fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(x, y, label="bpoly")
axs[0].scatter(
    torch.linspace(-10, 10, bpoly.order + 1),
    bpoly.theta.numpy().flatten(),
    label="theta",
)
axs[1].plot(x, adj, label="ajd")
# axs[1].scatter(
#     torch.linspace(-10, 10, bpoly.order),
#     bpoly.dtheta.numpy().flatten(),
#     label="dtheta",
# )
axs[1].plot(x, J, label="J")
fig.legend(ncols=4)
fig.tight_layout()
fig.show()

# %% with batch of data
M = 10
batch_size = 10
theta = torch.rand(size=(batch_size, M)) * 500  # creates a random parameter vector

# %% Test
bpoly = BernsteinTransform(theta=theta, linear=False)

x = torch.linspace(-11, 11, 2000).repeat(batch_size, 1)
y = bpoly(x)

adj = bpoly.log_abs_det_jacobian(x, y).detach()
J = torch.diag(torch.autograd.functional.jacobian(bpoly, x)).abs().log()

# %% bpoly draft
import torch


def binom_coef(n, k):
    # Use torch.lgamma for the logarithm of the gamma function
    log_gamma_n_plus_1 = torch.lgamma(torch.tensor(n + 1).float())
    log_gamma_k_plus_1 = torch.lgamma(torch.tensor(k + 1).float())
    log_gamma_n_minus_k_plus_1 = torch.lgamma(torch.tensor(n - k + 1).float())

    # Compute the binomial coefficient
    binom = torch.exp(
        log_gamma_n_plus_1 - log_gamma_k_plus_1 - log_gamma_n_minus_k_plus_1
    )

    return binom
