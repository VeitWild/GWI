import torch
import gpytorch
import math

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
