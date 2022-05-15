from abc import ABC,abstractmethod
import torch

class Prob_Metric_abstract(ABC):
    @abstractmethod
    def calculate_distance(self):
        pass

class Wasserstein_Distance(Prob_Metric_abstract):

    def __init__(self,GM1,GM2):
        self.GM1 = GM1
        self.GM2 = GM2
    def calculate_distance(self,X_S,X_B):

        N_S = X_S.size(0)
        N_B = X_B.size(0)

        #Mean Vectors
        m_P = self.GM1.mean(X_B)
        m_Q = self.GM2.mean(X_B)

        #Mean distance
        mean_dist = torch.square((m_P-m_Q)).mean()

        #Traces
        aver_trace_k=self.GM1.variance(X_B).mean() 
        aver_trace_r=self.GM2.variance(X_B).mean() #this is a bit inefficient since this trace could be calculated in the coviarance matrix call and returned jointly

        #Covariance Matrices
        k_XBXS = self.GM1.covariance_matrix(X_B,X_S)
        r_XSXB = self.GM2.covariance_matrix(X_S,X_B)

        #Eigenvalues
        eigenvalues = torch.linalg.eigvals(r_XSXB@k_XBXS+100*torch.eye(N_S)).real -100*torch.ones(N_S)

        eigenvalues[eigenvalues<0]=0

        #Wassersteindistance
        WD_squared = mean_dist + aver_trace_k + aver_trace_r - 2/((N_B*N_S)**0.5) * torch.sqrt(eigenvalues).sum()


        return WD_squared