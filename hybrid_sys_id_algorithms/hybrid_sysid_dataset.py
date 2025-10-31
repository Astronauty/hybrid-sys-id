from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

# class HybridSystemIdentificationDataset(Dataset):
#     """Dataset to process observation history for hybrid system identification."""
#     def __init__(self, observation_history: np.ndarray, num_legs: int, contact_force_dim: int, contact_force_start_idx: int):
#         """
#         Args:
#             observation_history (np.ndarray): Array of shape (num_samples, obs_dim)
#             num_legs (int): Number of legs in the robot (e.g., 4 for Ant)
#             contact_force_dim (int): Dimension of contact force per leg (e.g., 3 for 3D forces)
#         """
#         self.observation_history = observation_history
#         self.num_legs = num_legs
#         self.contact_force_dim = contact_force_dim
#         self.contact_force_start_idx = contact_force_start_idx

#         self.df = self.make_mode_label_dataframe()


#     def __len__(self):
#         return self.observation_history.shape[0]
    
#     def __getitem__(self):
#         pass

#     def contact_forces_to_mode_binary(self, contact_forces: np.ndarray) -> int:
#         """
#         Based on the lagrange multipliers (contact forces) in the observation, assign a contact mode label.
#         """
#         tol = 1e-3

#         binary_vec = np.zeros(self.num_legs, dtype=int)

#         for leg in range(self.num_legs):
#             start_idx = self.contact_force_start_idx + leg * self.contact_force_dim
#             end_idx = start_idx + self.contact_force_dim
#             force_vector = contact_forces[start_idx:end_idx]
#             # If any component of the contact force is positive, consider it in contact
#             if np.any(force_vector > tol):
#                 binary_vec[leg] = 1
#             else:
#                 binary_vec[leg] = 0
#         return binary_vec

#     def binary_vec_to_int(self, binary_vec: np.ndarray) -> int:
#         """Convert a binary vector (e.g., [0,1,0,1]) to an integer."""
#         return int("".join(str(int(binary_vec[i])) for i in range(len(binary_vec))), 2)
    
#     def make_mode_label_dataframe(self) -> pd.DataFrame:
#         """
#         Returns a pandas DataFrame with columns ['timestep', 'mode_label', 'mode_binary'].
#         """
#         mode_labels = []
#         mode_binaries = []
#         for t in range(len(self.observation_history)):
#             obs = self.observation_history[t]
#             contact_forces = obs.copy()  # assumes contact forces are in obs
#             binary_vec = self.contact_forces_to_mode_binary(contact_forces)
#             mode_label = self.binary_vec_to_int(binary_vec)
#             mode_labels.append(mode_label)
#             mode_binaries.append("".join(str(b) for b in binary_vec))
#         df = pd.DataFrame({
#             "timestep": np.arange(len(self.observation_history)),
#             "mode_label": mode_labels,
#             "mode_binary": mode_binaries
#         })
#         return df

from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class HybridSystemIdentificationDataset(Dataset):
    """Dataset to process observation history for hybrid system identification."""

    def __init__(self, observation_history: np.ndarray, num_legs: int, contact_force_dim: int, contact_force_start_idx: int):
        """
        Args:
            observation_history (np.ndarray): Array of shape (num_samples, obs_dim)
            num_legs (int): Number of legs in the robot (e.g., 4 for Ant)
            contact_force_dim (int): Dimension of contact force per leg (e.g., 3 for 3D forces)
            contact_force_start_idx (int): Index in obs where contact forces start
        """
        self.observation_history = observation_history
        self.num_legs = num_legs
        self.contact_force_dim = contact_force_dim
        self.contact_force_start_idx = contact_force_start_idx

        # Vectorized contact force extraction
        cf = self.observation_history[:, contact_force_start_idx : contact_force_start_idx + num_legs * contact_force_dim]
        self.contact_forces_matrix = cf.reshape(-1, num_legs, contact_force_dim)  # shape: (num_samples, num_legs, contact_force_dim)

        # Vectorized binary contact mode
        self.mode_binary_matrix = (np.any(self.contact_forces_matrix > 1e-3, axis=2)).astype(int)  # shape: (num_samples, num_legs)

        # Vectorized mode label
        powers = 2 ** np.arange(num_legs)[::-1]
        self.mode_labels = np.dot(self.mode_binary_matrix, powers)
        self.mode_binaries = ["".join(map(str, row)) for row in self.mode_binary_matrix]

        # Build DataFrame
        self.df = pd.DataFrame({
            "timestep": np.arange(len(self.observation_history)),
            "mode_label": self.mode_labels,
            "mode_binary": self.mode_binaries
        })

    def __len__(self):
        return self.observation_history.shape[0]

    def __getitem__(self, idx):
        obs = self.observation_history[idx]
        mode_label = self.mode_labels[idx]
        mode_binary = self.mode_binaries[idx]
        return {
            "observation": torch.tensor(obs, dtype=torch.float32),
            "mode_label": torch.tensor(mode_label, dtype = torch.long),
            "mode_binary": mode_binary
        }