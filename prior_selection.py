import torch
import gpytorch
import math
import sys
import tensorflow as tf
import gpflow
import numpy as np
from gpflow.utilities import print_summary
sys.path.append("../")
from RobustGP import robustgp


#init_method = robustgp.ConditionalVariance()
M = 10  # We choose 250 inducing variables
X = np.random.rand(1000, 2)
#print(X)
k = gpflow.kernels.SquaredExponential(variance=2.5,lengthscales=[0.1, 0.2])
print_summary(k)

# Initialise hyperparameters here
init_method = robustgp.ConditionalVariance()
Z = init_method.compute_initialisation(X, M, k)[0]
print(Z)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def initialise_prior(train_x,train_y,training_iter,kernel_prior,sigma2_0=1):

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood() #standard GPR likelihood

    likelihood.noise = sigma2_0


    model = ExactGPModel(train_x, train_y, likelihood,kernel=kernel_prior)

    ###Intialise the initialisation of prior hyperparameters
    

    hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1.),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
    'covar_module.outputscale': torch.tensor(2.),
    }

    model.initialize(**hypers)

    print(
        model.likelihood.noise_covar.noise.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.outputscale.item()
    )


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #    i + 1, training_iter, loss.item(),
        #    model.covar_module.base_kernel.lengthscale.item(),
        #    model.likelihood.noise.item()
        #))
        optimizer.step()

    return model.likelihood.noise.item()
