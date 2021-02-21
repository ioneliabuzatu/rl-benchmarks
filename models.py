import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines.common import make_vec_env
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise

from utils import save_model_weights


def ddpg(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None):
    from stable_baselines import DDPG

    env = gym.make(env_id)

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(policy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                 tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=timesteps, log_interval=log_interval)

    save_model_weights(model, "ddpg", env_id, policy)


def ppo2(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None):
    multiprocess_env = make_vec_env(env_id, n_envs=4)

    model = PPO2(policy, multiprocess_env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=timesteps, log_interval=log_interval)

    save_model_weights(model, "ppo2", env_id, policy)


def td3(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None):
    from stable_baselines.ddpg.noise import NormalActionNoise
    env = gym.make(env_id)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(policy, env, action_noise=action_noise, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=timesteps, log_interval=log_interval)

    save_model_weights(model, "td3", env_id, policy)


def sac(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None):
    env = gym.make(env_id)

    model = SAC(policy, env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=timesteps, log_interval=log_interval)

    save_model_weights(model, "sac", env_id, policy)


def trpo(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None):
    from stable_baselines import TRPO
    env = gym.make(env_id)

    model = TRPO(policy, env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=timesteps, log_interval=log_interval)

    save_model_weights(model, "trpo", env_id, policy)
