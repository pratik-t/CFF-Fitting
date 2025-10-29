import matplotlib.pyplot as plt
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# 1. Set up “data”
torch.manual_seed(0)
# three constant kinematic inputs
k = 5.0
Q2 = 2.0
x = 0.3
t = -0.172
# phi: 24 angles from 0 to 2π
phi = torch.linspace(-torch.pi, torch.pi, 24)

# “True” CFFs (unknown to the inference)
true_c = torch.tensor([-2.56464, 1.39564, 2.21195, 0.0315875])

# Dummy physics layer: 
#    dsig = c0*k + c1*Q2 + c2*x + c3*sin(phi)
def physics_layer(c, k, Q2, x, t, phi):
    return c[0] * k + c[1] * Q2 + c[2] * x + c[3] * torch.cos(phi) + t

# Generate noisy observations
sigma_true = physics_layer(true_c, k, Q2, x, t, phi)
sigma_err  = 0.1 * torch.ones_like(phi)
sigma_obs  = sigma_true + sigma_err * torch.randn_like(phi)

# 2. Define the Pyro model
def model(k, Q2, x, phi, obs, obs_err):
    # Weakly informative prior on 4 CFFs
    c = pyro.sample("c", dist.Normal(0., 10.).expand([4]).to_event(1))
    # Forward model
    sigma_pred = physics_layer(c, k, Q2, x, t, phi)
    # Likelihood
    pyro.sample("obs",
                dist.Normal(sigma_pred, obs_err).to_event(1),
                obs=obs)

# 3. Run MCMC with NUTS
pyro.clear_param_store()
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel,
            num_samples=200,
            warmup_steps=50,
            num_chains=1)
mcmc.run(k, Q2, x, phi, sigma_obs, sigma_err)

# 4. Extract and summarize posterior samples
posterior_samples = mcmc.get_samples()["c"]  # shape (1000, 4)
means = posterior_samples.mean(dim=0)
stds  = posterior_samples.std(dim=0)

for i, (m, s) in enumerate(zip(means, stds), start=1):
    print(f"CFF[{i}] posterior mean = {m:.3f},  std = {s:.3f}")


# Convert PyTorch tensors to NumPy arrays
posterior = posterior_samples.cpu().detach().numpy()  # shape (N, 4)
true_vals = true_c.cpu().numpy()                     # shape (4,)
phi_np = phi.cpu().numpy()                           # shape (24,)
obs_np = sigma_obs.cpu().numpy()                     # shape (24,)
err_np = sigma_err.cpu().numpy()                     # shape (24,)

# Numpy version of your physics layer


def physics_layer_np(c, k, Q2, x, phi):
    return c[0] * k + c[1] * Q2 + c[2] * x + c[3] * np.sin(phi)


# 1) Plot CFF posterior histograms with true values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.hist(posterior[:, i], bins=30, alpha=0.7)
    ax.axvline(true_vals[i], color='r', linestyle='--', label="True value")
    ax.set_title(f"CFF[{i+1}]")
    ax.legend()
plt.tight_layout()
plt.show()

# 2) Plot predicted dsig curves under the observed dsig with error bars
fig, ax = plt.subplots(figsize=(8, 5))
num_plot = min(200, posterior.shape[0])
for c in posterior[:num_plot]:
    sigma_pred_np = physics_layer_np(c, k, Q2, x, phi_np)
    ax.plot(phi_np, sigma_pred_np, alpha=0.1, color='blue')
ax.errorbar(phi_np, obs_np, yerr=err_np, fmt='o',
            color='black', label="Observed dsig")
ax.set_xlabel("phi")
ax.set_ylabel("dsig")
ax.legend()
plt.tight_layout()
plt.show()
