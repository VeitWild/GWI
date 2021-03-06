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
from sklearn.model_selection import train_test_split
from utils import make_deterministic

seed = 42
make_deterministic(seed)

#torch.autograd.set_detect_anomaly(True)

###     Simulate some data
N = 200
N_train = math.ceil(0.9 * N)


# M=math.ceil(N**0.5)
M = int(N_train**0.5)
M = 12
sigma_true = 0.5

X = np.linspace(-2, 2, N, dtype=np.float32).reshape(N, 1)
input_dim = X.shape[1]
Y = (
    (
        2 * np.sin(X * (2 * math.pi))
        + np.random.normal(loc=0.0, scale=sigma_true, size=(N, 1))
    )
    .reshape(N)
    .astype(np.float32)
)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=(N - N_train) / N, random_state=42
)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(Y_train)
Y_test = torch.from_numpy(Y_test)

### Subsample inducing points
indices = torch.ones(N_train).multinomial(
    num_samples=M, replacement=False
)  # samples random indices from 1:N
Z = X_train[
    indices,
]
Y_Z = Y_train[indices]

###Prior Measure
kernel_prior = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1]),
)
GM_prior = GM.GM_prior(gpytorch.means.ConstantMean(), kernel_prior)

###     Intiliase prior hyperparamters for training kernel with GPyTorch
training_iter = 100
sigma2_0 = gpytorch.likelihoods.GaussianLikelihood().noise

###     Prior Hyperparamters before training
# print('Prior Hyperparamters before training')
print("sigma", sigma2_0.item() ** 0.5)
print("kernel_lengthscale: ", kernel_prior.base_kernel.lengthscale.item())
print("kernel_outputscale: ", kernel_prior.outputscale.item())

###For now ok, but actually better to use intialise GM_prior and then handover the rest
###     Training with Gpytorch
# print(Z.size())
# print(Y_Z.size())

sigma2 = prior_selection.initialise_prior(
    train_x=Z,
    train_y=Y_Z,
    training_iter=training_iter,
    kernel_prior=kernel_prior,
    sigma2_0=sigma2_0,
)

###     Prior Hyperparamters after training
print("Prior Hyperparamters after training:")
print("sigma: ", sigma2**0.5)
print("kernel_lengthscale: ", GM_prior.kernel.base_kernel.lengthscale.item())
print("kernel_outputscale: ", GM_prior.kernel.outputscale.item())

# Initiliase Variational Measure
m_Q = meanfct.DNN(input_dim=input_dim)
GM_var = GM.GM_GWI_net(GM_prior=GM_prior, m_var=m_Q, landmark_points=Z)



# Intialse Covariance Matrix with batch of X
N_B = 100
N_S = 100
batch_indices = torch.ones(N_train).multinomial(num_samples=N_B, replacement=False)
X_batch = X_train[
    batch_indices,
]
Y_batch = Y_train[batch_indices]
X_S = X_train[
    torch.ones(N_train).multinomial(num_samples=N_S, replacement=False),
]

GM_var.initialise_L_matrix(X=X_batch, N=N_train, sigma2=sigma2)

# Calculate Wasserstein Distance
WD = probability_metrics.Wasserstein_Distance(GM_prior, GM_var)

# print(WD.calculate_distance(X_S=X_S,X_B=X_batch))

# print(GM_var.get_var_parameters())

###Probably nneed to concatenate all parameters now with
# print(GWI_loss.parameters())

# print(GWI_loss.calculate_loss(X_batch,Y_batch))
# print(GM_prior.mean(X))
# print(GM_var.variance(X))


# print(GM_var.covariance_matrix(X,X))

###Now the Training

GWI_loss = generalised_loss.GWI_regression_loss(GM_prior, GM_var, sigma2, N_train)

#for p in GM_var.parameters():
#   print(p)

opt = torch.optim.Adam(GM_var.parameters(), lr=0.1)


print("loss at start", GWI_loss.calculate_loss(X_batch, X_S, Y_batch))
epochs = 50

for epoch in range(epochs):
    for i in range((N_train - 1) // N_B + 1):
        start_i = i * N_B
        end_i = start_i + N_B
        xb = X_train[
            start_i:end_i,
        ]
        yb = Y_train[start_i:end_i]

        xs = X_train[
            torch.ones(N_train).multinomial(num_samples=N_S, replacement=False),
        ]

        opt.zero_grad()

        loss = GWI_loss.calculate_loss(xb, xs, yb)

        # print('loss loop:',loss)
        # print('i',i)

        #print(GM_var.L_diag_unconst)
        print(GM_var.L_lower)

        loss.backward()

        #print("weight_grad", GM_var.m_var.linear1.weight.grad)
        #print(GM_var.L_diag_unconst.grad)
        print(GM_var.L_lower.grad)
        opt.step()

        print("L_diag after step", GM_var.L_diag_unconst)
        print("L_lower after step", GM_var.L_lower)

print("loss in the end:", GWI_loss.calculate_loss(X_batch, X_S, Y_batch))
