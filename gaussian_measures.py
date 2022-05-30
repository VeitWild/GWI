from abc import ABC, abstractmethod
import gpytorch
import meanfct
import torch
import math
from utils import build_L_matrix

class GM_abstract(ABC):
    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def covariance_matrix(self):
        pass

    @abstractmethod
    def variance(self):
        pass


class GM_prior(GM_abstract):
    def __init__(self, meanfct, kernel):
        self.meanfct = meanfct
        self.kernel = kernel

    def mean(self, X):
        return self.meanfct(X)

    def covariance_matrix(self, X1, X2):
        return self.kernel(X1, X2).evaluate()

    def variance(self, X):
        return self.kernel(X, diag=True)


class GM_GWI_net(GM_abstract, torch.nn.Module):
    def __init__(self, GM_prior, m_var, landmark_points):

        super(GM_GWI_net, self).__init__()

        self.GM_prior = GM_prior
        self.m_var = m_var
        self.k_ZZ = self.GM_prior.covariance_matrix(
            landmark_points, landmark_points
        )  # cache at initiliasation
        # print(self.k_ZZ)
        self.M = self.k_ZZ.size(0)

        self.cholesky_k_ZZ = torch.linalg.cholesky(
            self.k_ZZ + 0.1 * torch.eye(self.M)
        )  # cache at initialisation
        self.landmark_points = landmark_points
        print('k_ZZ',self.k_ZZ)
        print('cholesky_k_ZZ',self.cholesky_k_ZZ)

        # Define Variational Paramters
        #self.cholesky_sigma_matrix = torch.nn.Parameter(
        #    torch.tril(torch.eye(self.M)), requires_grad=True
        #)  # maybe intiialise properly here already?

        # Define Variational Paramters
        self.L_diag_unconst = torch.nn.Parameter(
            torch.ones(self.M)
        )  # this is the exponent of the L diagonal matrix
        self.L_lower = torch.nn.Parameter(
            torch.ones(int(self.M * (self.M - 1) / 2))
        )  # entries of the lower part of the L matrix

    def mean(self, X):
        return self.GM_prior.mean(X) + self.m_var(X)

    def covariance_matrix(self, X1, X2):
        # Load paramters
        Z = self.landmark_points
        #M = Z.size(0)
        #lower = torch.tril(torch.ones(M, M))
        #L = self.cholesky_sigma_matrix * lower

        #Build Sigma matrix
        L= build_L_matrix(self.L_diag_unconst, self.L_lower, jitter=0.1)

        ## prior covariance matrices
        k_X1X2 = self.GM_prior.covariance_matrix(X1, X2)
        k_ZX1 = self.GM_prior.covariance_matrix(Z, X1)
        k_ZX2 = self.GM_prior.covariance_matrix(Z, X2)

        # (variational) posterior covariance
        inv = torch.cholesky_solve(k_ZX2, self.cholesky_k_ZZ)  # k_ZZ^{-1} k(Z,X2)
        prod1 = torch.t(k_ZX1) @ inv  # k(X1,Z)k_ZZ^{-1} k(Z,X2)

        prod2 = torch.t(k_ZX1) @ L  # k(X1,Z) L
        prod3 = torch.t(L) @ k_ZX2  # L^T k(Z,X2)


        return k_X1X2 - prod1 + prod2 @ prod3

    def variance(self, X):
        Z = self.landmark_points
        #M = Z.size(0)
        #lower = torch.tril(torch.ones(M, M))
        #L = self.cholesky_sigma_matrix * lower
        # print(L)
        L= build_L_matrix(self.L_diag_unconst, self.L_lower, jitter=0.01)

        # Calc diag k(X,X)
        diag1 = self.GM_prior.variance(X)

        # Calculate diagonal  k(X,Z) k_ZZ^{-1} k(Z,X)
        k_ZX = self.GM_prior.covariance_matrix(Z, X)
        inv = torch.cholesky_solve(k_ZX, self.cholesky_k_ZZ)  # k_ZZ^{-1} k(Z,X)
        diag2 = torch.mul(k_ZX, inv).sum(dim=0)

        # Calculate diagonal k(X,Z) Sigma k(Z,X)
        temp = torch.t(k_ZX) @ L  # NxM
        diag3 = torch.mul(temp, temp).sum(dim=1)

        return diag1 + diag2 + diag3

    def initialise_L_matrix(self, X, N, sigma2=1):
        Z = self.landmark_points
        M = Z.size(0)
        N_B = X.size(0)

        k_ZZ = self.GM_prior.covariance_matrix(Z, Z)
        k_ZX = self.GM_prior.covariance_matrix(Z, X)

        mat = k_ZZ + 1 / sigma2 * k_ZX @ torch.t(k_ZX) * N / N_B
        #print(mat)
        chol = torch.linalg.cholesky(mat+0.1*torch.eye(M))

        cholesky_sigma_matrix = torch.tril(torch.linalg.inv(chol)) # maybe you need to add jitter here
        #print('Sigma',cholesky_sigma_matrix)

        self.L_diag_unconst = torch.nn.Parameter( torch.log(torch.diag(cholesky_sigma_matrix)) )    

        indices = torch.tril_indices(M, M, -1)
        self.L_lower =  torch.nn.Parameter(cholesky_sigma_matrix[indices[0,],indices[1,]] )
       

       #self.L_lower = torch.nn.Parameter( torch.tril(cholesky_sigma_matrix, diagonal=-1).reshape(-1,) )
        #print('Lower',self.L_lower)
        






# Some tests for the classes

#kernel_prior = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

#mean_function = meanfct.DNN(input_dim=1)
