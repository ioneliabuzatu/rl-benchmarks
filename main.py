import logging
import sys

# noinspection PyUnresolvedReferences
import pybullet_envs

import wandb
from models import ddpg, sac, trpo, td3, ppo2
from utils import set_global_seed


def run_benchmarks(timesteps=400, single_model=False, single_model_name=None):
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    set_global_seed(123)

    envs_id = ["Pendulum-v0", "MountainCarContinuous-v0", "Hopper-v2", "Humanoid-v2", "Walker2DBulletEnv-v0",
               "HumanoidStandup-v2", "HalfCheetah-v2", "Swimmer-v2"]

    for env_name in envs_id:

        if single_model and single_model_name is not None:
            models[single_model_name](env_name, timesteps)
        else:
            for model_name, model_function in models.items():
                wandb.init(project="rl-benchmarks", reinit=True)
                wandb.run.name = f"{env_name}/{model_name}"
                logging.info(f"run name: {wandb.run.name}")

                model_function(env_name, timesteps)


models = {"ddpg": ddpg,
          "ppo2": ppo2,
          "sac": sac,
          "td3": td3,
          "trpo": trpo}

if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        if model_name in ["ddpg", "sac", "td3", "trpo", "ppo2"]:
            run_benchmarks(single_model=True, single_model_name=model_name)
    except IndexError:
        run_benchmarks()
