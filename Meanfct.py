import torch as torch

class DNN(torch.nn.Module):

    def __init__(self,input_dim):
        super(DNN, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 10)
        self.activation1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(10, 10)
        self.activation2 = torch.nn.Tanh()
        self.out = torch.nn.Linear(10,1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.out(x)

        return x