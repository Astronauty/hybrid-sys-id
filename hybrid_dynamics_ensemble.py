import torch
from torch import nn

class HybridDynamicsModeClassifier(nn.Module):
    def __init__(
        self,
        nx: int, 
        nu: int,
        nmodes: int,
        na: int, # Number of previous measurements to include in regressor
        nb: int, # Number of previous inputs to include in regressor
        nc: int, # Number of previous contact modes to include in regressor
        nlambda: int,
        nhidden: int = 64,
        nlayer: int = 2,
    ):
        super(HybridDynamicsModeClassifier, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_layers = [nn.Linear(nx*na + nu*nb + nlambda*nc, nhidden), nn.ReLU()]
        for _ in range(nlayer - 1):
            hidden_layers += [nn.Linear(nhidden, nhidden), nn.ReLU()]
        
        self.network = nn.Sequential(*hidden_layers, nn.Linear(nhidden, nmodes), nn.Softmax(dim=-1)).to(device)

    def forward(self, input):
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, nx + nu + nlambda)
        """
        return self.network(input)

class HybridDynamicsEnsemble(nn.Module):
    def __init__(
        self,
        nx: int, 
        nu: int,
        nmodes: int,
        nlambda: int,
        nhidden: int = 64,
        nlayer: int = 2,
    ):
        super(HybridDynamicsEnsemble, self).__init__()

        self.models = nn.ModuleList([
            HybridDynamicsModeClassifier(nx, nu, nmodes, nlambda, nhidden, nlayer)
            for _ in range(nmodes)
        ])

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, nx + nu + nlambda)
        """
        outputs = [model(input) for model in self.models]
        return torch.stack(outputs, dim=1)