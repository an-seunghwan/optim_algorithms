#%%
import numpy as np
import torch
#%%
config = {
    "n": 5000,
    "p": 10,
}
#%%
"""synthetic data setting"""
np.random.seed(1)
torch.random.manual_seed(1)

x = 2. * torch.rand(config["n"], config["p"]) - 1. # Uniform(-1, 1)
mean = torch.mean(x, axis=0)
std = torch.std(x, axis=0)
x = (x - mean) / std # normalize
true_beta = 2. * torch.rand(config["p"], 1) - 1. # Uniform(-1, 1)
epsilon = torch.randn(config["n"], 1)
y = torch.matmul(x, true_beta) + epsilon
#%%
def regression_function(x, y, beta):
    """
    f(beta) = 1/(2n) || y - x beta ||_2^2
    """
    return 0.5 * torch.mean(torch.sum(torch.pow(y - torch.matmul(x, beta), 2), axis=1))
#%%
"""convexity"""
beta1 = torch.ones(config["p"], 1) * 0.1
beta2 = torch.ones(config["p"], 1) * 0.9
t = 0.7
beta_inter = t * beta1 + (1. - t) * beta2

LHS = regression_function(x, y, beta_inter).item()
RHS = t * regression_function(x, y, beta1).item() + (1. - t) * regression_function(x, y, beta2).item()
print("convexity")
print("LHS: {}, RHS: {}".format(LHS, RHS))
assert LHS <= RHS
#%%
"""Lipschitz"""
hessian = torch.matmul(x.t(), x) / config["n"]
eigen_values, _ = torch.eig(hessian)
L = torch.max(eigen_values[:, 0])
m = torch.min(eigen_values[:, 0])
print("Lipschitz constant: {}".format(L))
print("strong convexity constant: {}".format(m))
assert (eigen_values[:, 0] > 0).all() # positive definite

beta1 = torch.ones(config["p"], 1) * 0.1
beta2 = torch.ones(config["p"], 1) * 0.9
beta1.requires_grad_(True)
beta2.requires_grad_(True)

regression1 = 0.5 * torch.mean(torch.sum(torch.pow(y - torch.matmul(x, beta1), 2), axis=1))
regression1.backward()
beta1_grad = beta1.grad
regression2 = 0.5 * torch.mean(torch.sum(torch.pow(y - torch.matmul(x, beta2), 2), axis=1))
regression2.backward()
beta2_grad = beta2.grad

LHS = torch.norm(beta1_grad - beta2_grad, p=2).item()
RHS = (L * torch.norm(beta1 - beta2, p=2)).item()
print("LHS: {}, RHS: {}".format(LHS, RHS))
assert LHS <= RHS
#%%
"""convergence analysis"""
k = 20
t = (1. / L).item() * 0.1 # step size, t <= 1/L
beta = torch.ones(config["p"], 1, requires_grad=True)
beta_grad = None

for iteration in range(k):
    loss = 0.5 * torch.mean(torch.sum(torch.pow(y - torch.matmul(x, beta), 2), axis=1))
    loss.backward()
    with torch.no_grad():
        beta_grad = beta.grad.clone()
        # beta = beta - t * beta.grad # Not work...
        beta -= t * beta.grad
        beta.grad.zero_()
    print("iteration: {} | loss: {}".format(iteration, round(loss.item(), 3)))
#%%
f_k = (0.5 * torch.mean(torch.sum(torch.pow(y - torch.matmul(x, beta), 2), axis=1))).detach().item()
f_optimal = (0.5 * torch.mean(torch.sum(torch.pow(y - torch.matmul(x, true_beta), 2), axis=1))).detach().item()
LHS = f_k - f_optimal
RHS = torch.norm(torch.ones(config["p"], 1) - true_beta, p=2) ** 2 / (2. * t * k)
print("convergence analysis")
print("LHS: {}, RHS: {}".format(LHS, RHS))
assert LHS <= RHS
#%%
eps = torch.pow(torch.norm(beta_grad, p=2), 2) / (2. * m)
# torch.sqrt(2. * m * eps).item()
# torch.norm(beta.grad, p=2).item()
print("stopping rule")
print("LHS: {}, eps: {}".format(LHS, eps))
assert LHS <= eps
#%%