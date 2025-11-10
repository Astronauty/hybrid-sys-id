import torch
from torch import nn
from hybrid_sysid_classifier import ModeClassifier
from environments.mujoco_gymnasium.contact_wrapped_antv5 import ContactForceWrapper


class HybridDynamicsModel(nn.Module):
    def __init__(
        self,
        n_modes: int, # Number of contact modes
        nx: int, # State dim
        na: int, # Number of previous states 
        nu: int, # Control dim
        nb: int, # Number of previous controls
        n_hiddenlayers: int,
        n_hiddenlayerdim: int,
    ):
        super(HybridDynamicsModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mode_classifier = ModeClassifier(nx*na + nu*nb, hidden_dim=64, num_hidden_layers=2, num_classes=n_modes).to(device)
        
        self.dynamics_submodels = nn.ModuleList([
             MLPDynamics(nx, na, nu, nb, nhiddenlayers=2, nhiddenlayerdim=64) for _ in range(n_modes)
        ])

    def forward(self, obs):
        """
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, nx + nu + nlambda)
        """
        predicted_mode_logits = self.mode_classifier(obs)
        mode_probs = torch.softmax(predicted_mode_logits, dim=-1)
        predicted_modes = torch.argmax(mode_probs, dim=-1)

        outputs = torch.zeros(obs.size(0), self.dynamics_submodels[0].net[-1].out_features, device=obs.device)

        for i, dyn_model in enumerate(self.dynamics_submodels):
            mask = (predicted_modes == i)
            if mask.any():
                outputs[mask] = dyn_model(obs[mask])


        return torch.stack(outputs, dim=1)


class MLPDynamics(nn.Module):
    def __init__(
            self,
            nx: int, # State dim
            na: int, # Number of previous states 
            nu: int, # Control dim
            nb: int, # Number of previous controls
            nhiddenlayers: int,
            nhiddenlayerdim: int,
    ):
            

            layers = []

            layers.append(nn.Linear(nx*na + nu*nb), nhiddenlayerdim)
            layers.append(nn.ReLU())

            for _ in range(nhiddenlayers):
                layers.append(nn.Linear(nhiddenlayerdim, nhiddenlayerdim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(nhiddenlayerdim, nx))

            self.net = nn.Sequential(*layers)

    def forward(self, input):
        outputs = self.net(input)
        #  return torch.stack(outputs, dim=1)
        return self.net(input)
    