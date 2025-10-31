import gymnasium as gym
import numpy as np

# Ant
geom_names = []

class ContactForceWrapper(gym.ObservationWrapper):
    def __init__(self, env, foot_geom_names):
        super().__init__(env)
        self._data = env.unwrapped.data
        self._model = env.unwrapped.model

        self.foot_body_ids = [self._model.geom(name).bodyid for name in foot_geom_names]

        # Original observation space
        orig_space = env.observation_space

        # num_forces = self._data.cfrc_ext.size

        # Each foot contributes 3D contact force (x, y, z)
        low = np.full(3 * len(self.foot_body_ids), -np.inf)
        high = np.full(3 * len(self.foot_body_ids), np.inf)

        # Augmented observation space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([orig_space.low, low]),
            high=np.concatenate([orig_space.high, high]),
            dtype=np.float64
        )

    def observation(self, obs):
        foot_forces = []
        # foot_locations = []

        for body_id in self.foot_body_ids:
            # External force on the body (world frame), first 3 entries are force
            force_vector = self._data.cfrc_ext[body_id].copy()

            force_vector = force_vector.flatten()[:3]  # Extract only the force component
            foot_forces.append(force_vector)

        foot_forces = np.concatenate(foot_forces).flatten()


        # transform_GRFs_to_world = True
        # if transform_GRFs_to_world:
        #     foot_forces = self._data.xmat @ foot_forces
        return np.concatenate([obs, foot_forces])
        # return obs

# env = gym.make("Ant-v5")

# foot_geom_names = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom"]
# env = ContactForceWrapper(env, foot_geom_names)
# obs, _ = env.reset()
# print("Observation shape:", obs.shape)

# for i in range(1000):
#     action = env.action_space.sample() * 0.0
#     obs, reward, terminated, truncated, info = env.step(action)

# # Now read foot forces
# print("Augmented observation (last 6 entries are foot forces):", obs[-12:])
# print(env.unwrapped.model.opt.timestep)
# print("Contacts:", len(env.unwrapped.data.contact))
# data = env.unwrapped.data
# model = env.unwrapped.model
# print("ncon:", data.ncon)
# for i in range(int(data.ncon)):
#     c = data.contact[i]
#     print(i, model.geom(c.geom1).name, "<->", model.geom(c.geom2).name)
# print("cfrc_ext shape:", data.cfrc_ext.shape)
# for i in range(model.nbody):
#     print(i, model.body(i).name, data.cfrc_ext[i])