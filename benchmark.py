import gym
import wandb
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

wandb.init()
wandb.tensorboard.patch(save=False, tensorboardX=True)

# Create the environment
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# Define the model
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./cartpolev1_tensorboard/")

# Train the agent
model.learn(total_timesteps=2500)



