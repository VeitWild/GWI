import torch
import numpy as np
import sklearn
import sys
import generalised_loss
import gaussian_measures as GM
import meanfct
import gpytorch
import math
import prior_selection
import probability_metrics

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

###     Simulate some data 
N=200
N_train = math.ceil(0.9*N)


#M=math.ceil(N**0.5)
M=int(N_train**0.5)
sigma_true=0.2

X = np.linspace(0,1,N).reshape(N,1)
input_dim = X.shape[1]
Y = (np.sin(X* (2 * math.pi)) + np.random.normal(loc=0.0, scale=sigma_true,size=(N,1))).reshape(N)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(N-N_train)/N, random_state=42)
X_train = torch.from_numpy(X_train); X_test = torch.from_numpy(X_test); Y_train = torch.from_numpy(Y_train); Y_test = torch.from_numpy(Y_test)



### Subsample inducing points
indices=torch.ones(N_train).multinomial(num_samples=M,replacement=False) # samples random indices from 1:N
Z = X_train[indices,]
Y_Z = Y_train[indices,]


###Prior Measure
kernel_prior=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]) )
GM_prior = GM.GM_prior(gpytorch.means.ConstantMean(),kernel_prior)

###     Intiliase prior hyperparamters for training kernel with GPyTorch
training_iter=50
sigma2_0 = gpytorch.likelihoods.GaussianLikelihood().noise

###     Prior Hyperparamters before training
#print('Prior Hyperparamters before training')
#print('sigma',sigma2_0.item()**0.5)
#print('kernel_lengthscale: ',kernel_prior.base_kernel.lengthscale.item())
#print('kernel_outputscale: ',kernel_prior.outputscale.item())

###For now ok, but actually better to use intialise GM_prior and then handover the rest
###     Training with Gpytorch
#print(Z.size())
#print(Y_Z.size())

sigma2=prior_selection.initialise_prior(train_x=Z,train_y=Y_Z,training_iter=training_iter,kernel_prior=kernel_prior,sigma2_0=sigma2_0)

###     Prior Hyperparamters after training
print('Prior Hyperparamters after training:')
print('sigma: ',sigma2**0.5)
print('kernel_lengthscale: ',GM_prior.kernel.base_kernel.lengthscale.item())
print('kernel_outputscale: ',GM_prior.kernel.outputscale.item())

#Initiliase Variational Measure
m_Q = meanfct.DNN(input_dim=input_dim)
GM_var = GM.GM_GWI_net(GM_prior=GM_prior,m_var=m_Q,landmark_points=Z)

#Intialse Covariance Matrix with batch of X
N_B= 100
N_S= 100
batch_indices = torch.ones(N_train).multinomial(num_samples=N_B,replacement=False)
X_batch = X_train[batch_indices,]
Y_batch = Y_train[batch_indices]
X_S = X[torch.ones(N_train).multinomial(num_samples=N_S,replacement=False),]

GM_var.initialise_Sigma_matrix(X=X_batch,N=N,sigma2=sigma2)

#Calculate Wasserstein Distance
WD = probability_metrics.Wasserstein_Distance(GM_prior,GM_var)

#print(WD.calculate_distance(X_S=X_S,X_B=X_batch))

GWI_loss = generalised_loss.GWI_regression_loss(GM_prior,GM_var,sigma2,N,N_S)

for p in m_Q.parameters():
    print(p)

#print(GWI_loss.parameters())

#print(GWI_loss.calculate_loss(X_batch,Y_batch))
#print(GM_prior.mean(X))
#print(GM_var.variance(X))
#print(GM_var.mean(X))
#print(GM_var.covariance_matrix(X,X))

###Now the Training

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // N_B + 1):
        start_i = i * N_B
        end_i = start_i + N_B
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = GWI_loss(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(GWI_loss(model(xb), yb))





