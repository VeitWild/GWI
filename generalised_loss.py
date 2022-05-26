import numpy
import torch
import gpytorch
import probability_metrics


class GWI_regression_loss:
    def __init__(self, GM_prior, GM_var, sigma2, N):

        self.GM_prior = GM_prior  # you kinda dont need this
        self.GM_var = GM_var  # you kinda dont need this
        self.sigma2 = torch.tensor(sigma2)  #
        self.N = N
        self.WD = probability_metrics.Wasserstein_Distance(GM_prior, GM_var)

    def calculate_loss(self, X_B, X_S, Y_B):

        N_B = X_B.size(0)
        N_S = X_S.size(0)

        # get all the components to calculate WD
        m_P, m_Q, aver_trace_k, aver_trace_r, eigenvalues = self.WD.calculate_distance(
            X_S, X_B, components=True
        )

        # Calculate Wasserstein Distance
        mean_dist = torch.square((m_P - m_Q)).mean()
        WD_squared = (
            mean_dist
            + aver_trace_k
            + aver_trace_r
            - 2 / ((N_B * N_S) ** 0.5) * torch.sqrt(eigenvalues).sum()
        )

        # Calculate Expected Log-Likelihood
        const = (
            self.N / 2 * torch.log(2 * 3.14159 * self.sigma2)
        )  # actually irrelvant but maybe at some point relevant
        pred_error = torch.square(Y_B - m_P).mean()

        print("pred_err", pred_error)
        print("trace_r", aver_trace_r)
        ell = const + self.N / (2 * self.sigma2) * (pred_error + aver_trace_r)

        loss = ell + WD_squared  # maybe square-rrot WD_squared but kinda irrelevant

        print("ell", ell)
        print("WD2", WD_squared)

        return loss
