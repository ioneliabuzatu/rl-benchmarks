import gym
import wandb
import numpy as np
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines.common import make_vec_env
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from callbacks import WandbRenderEnvCallback

# noinspection PyUnresolvedReferences
from utils import save_model_weights


def ddpg(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None, seed=None, load_weights=None):
    from stable_baselines import DDPG

    env = gym.make(env_id)

    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    if load_weights is not None:
        model = DDPG.load(load_weights, env=env)
    else:
        model = DDPG(policy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log=tensorboard_log)

    callback = WandbRenderEnvCallback(model_name="ddpg", env_name=env_id)

    model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback)
    save_model_weights(model, "ddpg", env_id, policy, seed=seed, path=".")


def ppo2(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None, seed=None, load_weights=None):
    multiprocess_env = make_vec_env(env_id, n_envs=4)

    if load_weights is not None:
        model = PPO2.load(load_weights, multiprocess_env, verbose=0)
    else:
        model = PPO2(policy, multiprocess_env, verbose=1, tensorboard_log=tensorboard_log)

    callback = WandbRenderEnvCallback(model_name="ppo2", env_name=env_id)

    model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback)

    # save_model_weights(model, "ppo2", env_id, policy, seed, path=".")


def td3(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None, seed=None, load_weights=None):
    from stable_baselines.ddpg.noise import NormalActionNoise
    env = gym.make(env_id)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if load_weights is not None:
        model = TD3.load(load_weights, env, verbose=0)
    else:
        model = TD3(policy, env, action_noise=action_noise, verbose=1, tensorboard_log=tensorboard_log)

    callback = WandbRenderEnvCallback(model_name="td3", env_name=env_id)

    model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback)

    # save_model_weights(model, "td3", env_id, policy, seed)


def sac(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None, seed=None, load_weights=None):
    env = gym.make(env_id)

    if load_weights is not None:
        model = SAC.load(load_weights, env, verbose=0)
    else:
        model = SAC(policy, env, verbose=1, tensorboard_log=tensorboard_log)

    callback = WandbRenderEnvCallback(model_name="sac", env_name=env_id)

    model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback)

    # save_model_weights(model, "sac", env_id, policy, seed)


def trpo(env_id, timesteps, policy="MlpPolicy", log_interval=None, tensorboard_log=None, seed=None, load_weights=None):
    from stable_baselines import TRPO
    env = gym.make(env_id)

    if load_weights is not None:
        model = TRPO.load(load_weights, env=env, verbose=0)
    else:
        model = TRPO(policy, env, verbose=1, tensorboard_log=tensorboard_log)

    callback = WandbRenderEnvCallback(model_name="trpo", env_name=env_id)

    model.learn(total_timesteps=timesteps, log_interval=log_interval, callback=callback)

    # save_model_weights(model, "trpo", env_id, policy, seed)
