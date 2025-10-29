import wandb
from stable_baselines3 import PPO
import gymnasium as gym

wandb.init(project="ant-ppo", sync_tensorboard=True, name="ppo_ant_run")


logdir = "./logs/ant_ppo"
env_train = gym.make("Ant-v5")
model = PPO("MlpPolicy", env_train, verbose=1, tensorboard_log=logdir, batch_size=2048, n_steps=2048, learning_rate=3e-4)
model.learn(total_timesteps=100_000)  # you can increase timesteps for better policy
env_train.close()

wandb.finish()