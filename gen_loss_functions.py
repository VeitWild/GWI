import numpy
import torch
import gpytorch


class GWI_loss_regression():

   def __init__(self,X,Y,GM_prior,GM_var,sigma2):
      self.X = X
      self.Y = Y
      self.GM_prior = GM_prior
      self.GM_var = GM_var
      self.sigma2 = sigma2
      
   def calculate_loss(self):



      return


