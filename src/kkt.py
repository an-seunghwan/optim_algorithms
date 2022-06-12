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

Q_ = torch.randn(config["p"], config["p"])
Q = Q_.t() @ Q_
eigen_values, _ = torch.eig(Q)
pos = (eigen_values[:, 0] > 0).all()
print("Positive definite matrix:", pos.item())

c = 2. * torch.rand(config["p"], 1) - 1.
m = 5 # number of equality constraints
A = 2. * torch.rand(m, config["p"]) - 1.
b = 2. * torch.rand(m, 1) - 1.
#%%
"""KKT system"""
tmp = torch.linalg.inv(torch.cat([torch.cat([Q, A.t()], axis=1),
                      torch.cat([A, torch.zeros(m, m)], axis=1)]))
solution = tmp @ torch.cat([-c, b], axis=0)
x = solution[:config["p"], :]
#%%
"""check"""
opt_value = 0.5 * x.t() @ Q @ x + c.t() @ x
print("Optimal value: {}".format(opt_value.item()))
constraint = ((A @ x - b) <= 1e-6).all()
print("Satisfy constraint:", constraint.item())
#%%