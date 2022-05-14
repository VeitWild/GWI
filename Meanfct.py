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

DNN_reg = DNN(input_dim=5)

print('The model:')
print(DNN_reg)

print('\n\nJust one layer:')
print(DNN_reg.linear2)

print('\n\nModel params:')
for param in DNN_reg.parameters():
    print(param)

print('\n\nLayer params:')
for param in DNN_reg.linear2.parameters():
    print(param)


print(DNN_reg(torch.rand(100,5)))

torch.rand(2,4,5)