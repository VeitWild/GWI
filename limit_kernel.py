import numpy as np
import matplotlib.pyplot as plt

class Kernel_Limit(object):
    def __init__(self, nr_hidden_layers,gamma_weights,gamma_bias,activation_fct,N_S=100):
        self.L = nr_hidden_layers
        self.gamma_weights = gamma_weights
        self.gamma_bias = gamma_bias
        self.activation_fct = activation_fct
        self.N_S = N_S # Number of samples used to approximate integral
    
    def calculate_covariance(self,X1,X2): 

        #X1 NxD
        #X2 N'xD

        alpha= self.gamma_weights
        beta = self.gamma_bias
        D = X1.shape[1]
        N = X1.shape[0]
        phi = self.activation_fct
        N_S = self.N_S

        Sigma_0 = 1/D * alpha[0]*X1 @ X2.T + beta[0]

        Sigma_l = Sigma_0
        

        for l in range(1,self.L+1):

            eigs, V =np.linalg.eigh(Sigma_l+np.eye(N))
            #print(eigs)
            eigs[eigs<1] = 1
            eigs = eigs -1  
            #print(eigs)

            Sigma_sqrt = V * np.sqrt(eigs) # NxN
            std_normals = np.random.normal(0,1,(N,N_S)) # NxN_S
            cor_normals = Sigma_sqrt @ std_normals # N x N_S

            trafo_normals = phi(cor_normals) # N x N_S

            outer_matrix = np.einsum('ib,jb->ijb',trafo_normals,trafo_normals) # N x N x N_S contains phi(N_xn) x phi(N_xn') x b

            Sigma_l =  alpha[l]*outer_matrix.mean(axis=2) + beta[l]
        
        return Sigma_l

    def calculate_covariance2(self,X1,X2):
        #Uses seperate sample for each entry
        #should not make a difference
        #X1 NxD
        #X2 N'xD

        alpha= self.gamma_weights
        beta = self.gamma_bias
        D = X1.shape[1]
        N = X1.shape[0]
        phi = self.activation_fct
        N_S = self.N_S

        Sigma_0 = 1/D * alpha[0]*X1 @ X2.T + beta[0]

        Sigma_l = Sigma_0

        Sigma_small_0 = np.array([[Sigma_0[0,0],Sigma_0[0,1]],[Sigma_0[1,0],Sigma_0[1,1]]])

        for n in range(0,N):
            for n_p in range(n,N):
                Sigma_small_0 = np.array([[Sigma_0[n,n],Sigma_0[n,n_p]],[Sigma_0[n,n_p],Sigma_0[n_p,n_p]]])
                Sigma_small_l = Sigma_small_0

                for l in range(1,self.L+1):
                        
                    eigs, V =np.linalg.eigh(Sigma_small_l+np.eye(2))

                    eigs[eigs<1] = 1
                    eigs = eigs-1
                    #print(eigs)

                    Sigma_sqrt = V * np.sqrt(eigs) # 2x2
                    std_normals = np.random.normal(0,1,(2,N_S)) # 2xN_S
                    cor_normals = Sigma_sqrt @ std_normals #2xN_S
                    #print(cor_normals)

                    trafo_normals = phi(cor_normals) # 2xN_S
                    var_xn = np.square(trafo_normals[0,:]).mean()
                    var_xnp = np.square(trafo_normals[1,:]).mean()
                    cov_n_np = np.prod(trafo_normals,axis=0).mean()
                    #print(cov_n_np)

                    #Samples New for the variances
                    #Didn't make a difference here if you sample the variances or not
                    #Z1 = np.random.normal(0,1,N_S)
                    #Z2 = np.random.normal(0,1,N_S)
                    #N_xn = Sigma_sqrt[0,0]*Z1 + Sigma_sqrt[0,1]*Z2
                    #N_xnp = Sigma_sqrt[1,0]*Z1 + Sigma_sqrt[1,1]*Z2

                    #var_xn =   np.square(phi(N_xn)).mean()
                    #var_xnp =  np.square(phi(N_xnp)).mean()

                    Sigma_small_l = alpha[l]* np.array([[var_xn,cov_n_np],[cov_n_np,var_xnp]]) + beta[l]
                    Sigma_l[n,n_p] = cov_n_np

        

        

            #for l in range(1,self.L+1):



          #  eigs, V =np.linalg.eigh(Sigma_l+np.eye(N))
            #print(eigs)
           # eigs[eigs<1] = 1
            #eigs = eigs -1  
            #print(eigs)

            #Sigma_sqrt = V * np.sqrt(eigs) # NxN
            #std_normals = np.random.normal(0,1,(N,N,N_S,2)) #NxNxN_Sx2
            #cor_normals = Sigma_sqrt.reshape(1,N,N) @ std_normals # NxN x N_Sx2

            #trafo_normals = phi(cor_normals) # NxN x N_S
            #print('trafo_normals',trafo_normals.shape)
            
            #cov = trafo_normals.mean(axis=2) # NxN

            #outer_matrix = np.einsum('ib,jb->ijb',trafo_normals,trafo_normals) # N x N x N_S contains phi(N_xn) x phi(N_xn') x b
            #print('outer',outer_matrix.shape)

        # Sigma_l =  alpha[l]*cov + beta[l]
        
        return Sigma_l




    def sample_function(self,X,rough=False):

        N = X.shape[0]
        if rough==True:
            Sigma_l = self.calculate_covariance2(X,X)
        else:
            Sigma_l = self.calculate_covariance(X,X)

        eigs, V =np.linalg.eigh(Sigma_l+np.eye(N),UPLO='U')

        eigs[eigs<1] = 1
        eigs = eigs-1

        Sigma_sqrt = V * np.sqrt(eigs)

        std_normals = np.random.normal(0,1,(N,1))

        f_X = Sigma_sqrt @ std_normals # N x 1

        return f_X
    

    def plot_function(self,X,rough=False):

        f_X = self.sample_function(X,rough)

        plt.plot(X,f_X)
        plt.show()


    
def sigmoid(x):
    return 1/(1+np.exp(-x)) 


# Test the kernel
L = 2
N=  100
N_S = 300

gamma_weights = np.ones(L+2)
gamma_bias = np.ones(L+2)

k  = Kernel_Limit(nr_hidden_layers=L,gamma_weights=gamma_weights,gamma_bias=gamma_bias,activation_fct=sigmoid,N_S=N_S)

X= np.linspace(0,1,N).reshape(-1,1)
Z = np.linspace(5,10,N).reshape(-1,1)

#print(k.calculate_covariance2(X,X))

k.plot_function(X,rough=True)
#k.plot_function(X,rough=True)



#print(Sigma)
#A = np.prod(np.arange(1,13).reshape(3,2,2))

#B = np.arange(1,5).reshape(1,2,2)


#print(A @ B)

#print(np.einsum('ijk,ikl->ijl', A, B))
#Broad Casting Matmul






