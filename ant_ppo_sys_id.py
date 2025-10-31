import wandb
from stable_baselines3 import PPO
import gymnasium as gym

wandb.init(project="ant-ppo", sync_tensorboard=True, name="ppo_ant_run")

# Custom callback to log observation history
class ObservationHistoryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ObservationHistoryCallback, self).__init__(verbose)
        self.observation_history = []

    def _on_step(self) -> bool:
        # Access the current observation
        obs = self.locals["obs"]
        self.observation_history.append(obs)
        return True

    def get_observation_history(self):
        return np.array(self.observation_history)

logdir = "./logs/ant_ppo"
env_train = gym.make("Ant-v5")
model = PPO("MlpPolicy", env_train, verbose=1, tensorboard_log=logdir, batch_size=2048, n_steps=2048, learning_rate=3e-4)
# model.learn(total_timesteps=100_000)  # you can increase timesteps for better policy
model.learn(total_timesteps=10)  # you can increase timesteps for better policy

env_train.close()

model.save("models/ppo_antv5")



wandb.finish()