import logging
import sys

# noinspection PyUnresolvedReferences
import pybullet_envs

from utils import init_wandb_run
import wandb
from models import ddpg, sac, trpo, td3, ppo2
from utils import set_global_seed

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def run_benchmarks(time_steps=4000, single_model_name=None, single_env_name=None, project_name="rl-benchmarks", run_tag="mlp"):
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    set_global_seed(123)

    envs_id = ["Pendulum-v0", "ReacherBulletEnv-v0", "Hopper-v2", "Humanoid-v2", "HalfCheetah-v2", "HumanoidStandup-v2"]

    if single_env_name is not None and single_model_name is not None:
        init_wandb_run(project_name, single_env_name, single_model_name, f"{single_env_name}/{single_model_name}/{run_tag}")
        models[single_model_name](single_env_name, time_steps)

    else:
        for env_name in envs_id:

            if single_model_name is not None:
                init_wandb_run(project_name, env_name, single_model_name, f"{env_name}/{single_model_name}/{run_tag}")
                models[single_model_name](env_name, time_steps)

            else:
                for model_name, model_function in models.items():
                    init_wandb_run(project_name, env_name, model_name, f"{env_name}/{model_name}/{run_tag}")
                    model_function(env_name, time_steps)


models = {"ddpg": ddpg,
          "ppo2": ppo2,
          "sac": sac,
          "td3": td3,
          "trpo": trpo}

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        model_name = sys.argv[1]
        if model_name in ["ddpg", "sac", "td3", "trpo", "ppo2"]:

            if len(args) == 3:
                env_name = sys.argv[2]
                run_benchmarks(single_model_name=model_name, single_env_name=env_name)
            else:
                run_benchmarks(single_model_name=model_name, project_name="test")

    else:
        run_benchmarks()
