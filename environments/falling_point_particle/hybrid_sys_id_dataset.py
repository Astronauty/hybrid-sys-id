import torch
from torch.utils.data import Dataset
import numpy as np

class HybridSysIDDataset(Dataset):
    """PyTorch dataset for hybrid system identification with time series data.
    
    Creates feature vectors from past states and targets from next states.
    """
    
    def __init__(self, x, na=1, transform=None):
        """Initialize the dataset.
        
        Args:
            x (np.ndarray): State data of shape (N, 4) where columns are:
                           [x_pos, y_pos, x_vel, y_vel]
            na (int): Number of past states to include in feature vector
            transform (callable, optional): Optional transform to apply to data
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.na = na
        self.transform = transform
        
        # Number of valid samples (we lose na samples at the beginning)
        self.n_samples = len(x) - na
        
        if self.n_samples <= 0:
            raise ValueError(f"na={na} is too large for data length {len(x)}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Get a sample (features, target) pair.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (features, target) where:
                - features: tensor of shape (na * 4,) containing past states
                - target: tensor of shape (4,) containing next state
        """
        # Ensure idx is within valid range
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")
        
        # Get past states for features (from idx to idx+na)
        past_states = self.x[idx:idx+self.na]  # Shape: (na, 4)
        features = past_states.flatten()       # Shape: (na * 4,)
        
        # Get next state as target
        target = self.x[idx + self.na]         # Shape: (4,)
        
        if self.transform:
            features = self.transform(features)
            target = self.transform(target)
        
        return features, target
    
    def get_feature_names(self):
        """Get descriptive names for features."""
        state_names = ['x_pos', 'y_pos', 'x_vel', 'y_vel']
        feature_names = []
        
        for lag in range(self.na, 0, -1):  # From na steps back to 1 step back
            for state in state_names:
                feature_names.append(f"{state}_lag_{lag}")
        
        return feature_names
    
    def get_target_names(self):
        """Get descriptive names for targets."""
        return ['x_pos_next', 'y_pos_next', 'x_vel_next', 'y_vel_next']
    
    def get_raw_data(self):
        """Get the raw state data."""
        return self.x.numpy()
