import gymnasium as gym
from typing import Optional
from hybrid_system_sim import HybridSystemSim

"""

State:
    Fully Observable:
        - Continuous state (positions, velocities, lagrange multipliers)
        - Discrete state (contact mode)

    Partially Observable (No contacts):
        - Continuous state (positions, velocities)
        - Discrete state (contact mode)

Action:
    - Control inputs (forces/torques)


Step: 
"""

class HybridDynamicsEnv(gym.Env):
    def __init__(self, hybrid_sim: HybridSystemSim):
        super(HybridDynamicsEnv, self).__init__()
        self.hybrid_sim = hybrid_sim

    def step(self, action):
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        pass

