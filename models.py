import gym
import numpy as np
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines import TRPO
from stable_baselines.common import make_vec_env
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.noise import NormalActionNoise

from utils import save_model_weights


def ddpg(env_id, timesteps):
    env = gym.make(env_id)

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                 tensorboard_log="./tensorboard-logs/ddpg_mountaincar")
    DDPG.load(f"./weights/ddpg_test")
    model.learn(total_timesteps=timesteps)

    model.save(f"./weights/ddpg_test")
    # save_model_weights(model, "ddpg", env_id)


def ppo2(env_id, timesteps):
    multiprocess_env = make_vec_env(env_id, n_envs=4)

    model = PPO2("MlpPolicy", multiprocess_env, verbose=1, tensorboard_log=f"./tensorboard-logs/{env_id}")
    model.learn(total_timesteps=timesteps)

    save_model_weights(model, "ppo2", env_id)


def td3(env_id, timesteps):
    env = gym.make(env_id)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=f"./tensorboard-logs/{env_id}")
    model.learn(total_timesteps=timesteps)

    save_model_weights(model, "td3", env_id)


def sac(env_id, timesteps):
    env = gym.make(env_id)

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard-logs/{env_id}")
    model.learn(total_timesteps=timesteps)

    save_model_weights(model, "sac", env_id)


def trpo(env_id, timesteps):
    env = gym.make(env_id)

    model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard-logs/{env_id}")
    model.learn(total_timesteps=timesteps)

    save_model_weights(model, "trpo", env_id)
