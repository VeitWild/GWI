from abc import ABC,abstractmethod
from statistics import mean
import gpytorch
import meanfct
import torch
import math

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

    def __init__(self,meanfct,kernel):
        self.meanfct = meanfct
        self.kernel = kernel
    
    def mean(self,X):
        return self.meanfct(X)
    
    def covariance_matrix(self,X1,X2):
        return self.kernel(X1,X2).evaluate()
    
    def variance(self,X):
        return self.kernel(X,diag=True)

class GM_GWI_net(GM_abstract):
    def __init__(self,GM_prior,m_var,landmark_points):
        self.GM_prior = GM_prior
        self.m_var = m_var
        self.k_ZZ = self.GM_prior.covariance_matrix(landmark_points,landmark_points) #cache at initiliasation
        print(self.k_ZZ)

        self.cholesky_k_ZZ = torch.linalg.cholesky(self.k_ZZ) #cache at initialisation        
        self.landmark_points = landmark_points

        #Define Variational Paramters
        self.cholesky_sigma_matrix = torch.nn.Parameter(torch.tril(torch.eye(landmark_points.size(0)))) #maybe intiialise properly here already?
    
    def mean(self,X):
        return self.GM_prior.mean(X) + self.m_var(X)
    
    def covariance_matrix(self,X1,X2):
        Z = self.landmark_points

        ## prior covariance matrices
        k_X1X2 = self.GM_prior.covariance_matrix(X1,X2)
        k_ZX1  = self.GM_prior.covariance_matrix(Z,X1)
        k_ZX2  = self.GM_prior.covariance_matrix(Z,X2)


        #(variational) posterior covariance
        inv = torch.cholesky_solve(k_ZX2,self.cholesky_k_ZZ)      #k_ZZ^{-1} k(Z,X2)
        prod1 = torch.t(k_ZX1)@inv                          #k(X1,Z)k_ZZ^{-1} k(Z,X2)

        prod2 = torch.t(k_ZX1)@self.cholesky_sigma_matrix    # k(X1,Z) L
        prod3=  torch.t(self.cholesky_sigma_matrix)@k_ZX2    #L^T k(Z,X2)

        return k_X1X2 - prod1 + prod2@prod3
    
    def variance(self,X):
        Z = self.landmark_points

        #Calc diag k(X,X)
        diag1 = self.GM_prior.variance(X)

        #Calculate diagonal  k(X,Z) k_ZZ^{-1} k(Z,X)
        k_ZX = self.GM_prior.covariance_matrix(Z,X)
        inv = torch.cholesky_solve(k_ZX,self.cholesky_k_ZZ)  #k_ZZ^{-1} k(Z,X)
        diag2 = torch.mul(k_ZX,inv).sum(dim=0)

        #Calculate diagonal k(X,Z) Sigma k(Z,X)
        temp = torch.t(k_ZX)@self.cholesky_sigma_matrix #NxM
        diag3 = torch.mul(temp,temp).sum(dim=1)

        return diag1 + diag2 + diag3

    def initialise_Sigma_matrix(self,X,N,sigma2=1):
        Z = self.landmark_points
        N_B = X.size(0)

        k_ZZ = self.GM_prior.covariance_matrix(Z,Z)
        k_ZX = self.GM_prior.covariance_matrix(Z,X)

        mat = k_ZZ + 1/sigma2 * k_ZX @ torch.t(k_ZX) *N/N_B
        chol = torch.linalg.cholesky(mat) 

        self.cholesky_sigma_matrix = torch.tril(torch.linalg.inv(chol))
        self.cholesky_sigma_matrix.requires_grad_()





    
#Some tests for the classes

kernel_prior =gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

mean_function = meanfct.DNN(input_dim=1)







