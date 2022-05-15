import torch
import sys
import gen_loss_functions
import GaussianMeasures as GM
import meanfct
import gpytorch
import math
import prior_selection
import probability_metrics

###     Simulate some data 
N=100
M=math.ceil(N**0.5)
sigma_true=0.1
X = torch.linspace(-3,3,steps=N).reshape(N,1) #create 250 equidistand data points
input_dim = X.size(1)

Y = (torch.sin(X)+torch.cos(X**2)).reshape(N,)+ sigma_true*torch.randn(N)

###     Initialise Kernel
kernel_prior=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) )

###     Intiliase prior hyperparamters for training kernel with GPyTorch
training_iter=5
sigma2_0 = gpytorch.likelihoods.GaussianLikelihood().noise

###     Prior Hyperparamters before training
print('Prior Hyperparamters before training')
print('sigma',sigma2_0.item()**0.5)
print('kernel_lengthscale: ',kernel_prior.base_kernel.lengthscale.item())
print('kernel_outputscale: ',kernel_prior.outputscale.item())

###     Training with Gpytorch
sigma2=prior_selection.initialise_prior(train_x=X,train_y=Y,training_iter=training_iter,kernel_prior=kernel_prior,sigma2_0=sigma2_0)

###     Prior Hyperparamters after training
print('Prior Hyperparamters after training:')
print('sigma: ',sigma2**0.5)
print('kernel_lengthscale: ',kernel_prior.base_kernel.lengthscale.item())
print('kernel_outputscale: ',kernel_prior.outputscale.item())

#Initiliase variational paramters
indices=torch.ones(N).multinomial(num_samples=M,replacement=False) # samples random indices from 1:N
Z = X[indices,]
mean_function = meanfct.DNN(input_dim=input_dim)

#Intiliase Prior and Variational Measure
GM_prior = GM.GM_prior(mean_function,kernel_prior)
GM_var = GM.GM_GWI_net(GM_prior=GM_prior,m_var=mean_function,landmark_points=Z)


#Intialse Covariance Matrix with batch of X
N_B= 100
N_S= 100
X_batch = X[torch.ones(N).multinomial(num_samples=N_B,replacement=False),]
X_S = X[torch.ones(N).multinomial(num_samples=N_S,replacement=False),]

GM_var.initialise_Sigma_matrix(X=X_batch,N=N,sigma2=sigma2)

#Calculate Wasserstein Distance
WD = probability_metrics.Wasserstein_Distance(GM_prior,GM_var)

print(WD.calculate_distance(X_S=X_S,X_B=X_batch))



#print(GM_prior.variance(X))
#print(GM_var.variance(X))
#print(GM_var.mean(X))
#print(GM_var.covariance_matrix(X,X))






