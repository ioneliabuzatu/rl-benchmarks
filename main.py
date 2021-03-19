import logging
import os
import sys

# noinspection PyUnresolvedReferences
import pybullet_envs

from utils import init_wandb_run
import wandb
from models import ddpg, sac, trpo, td3, ppo2
from utils import set_global_seed

logging.getLogger("tensorflow").setLevel(logging.ERROR)
api = wandb.Api()


def run_benchmarks(time_steps=4000, single_model_name=None, single_env_name=None, project_name="rl-benchmarks",
                   run_tag="mlp", log_interval=1000, tensorboard_log="./tensorboard-logs", seed=123,
                   policy_type="MlpPolicy"):
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    set_global_seed(seed)

    envs_id = ["Pendulum-v0", "ReacherBulletEnv-v0", "Hopper-v2", "Humanoid-v2", "", "HumanoidStandup-v2", "HalfCheetah-v2"]

    if single_env_name is not None and single_model_name is not None:
        init_wandb_run(
            project_name, single_env_name, single_model_name, f"{single_env_name}/{single_model_name}-{run_tag}",
            dir="."
        )

        models[single_model_name](
            single_env_name, time_steps, log_interval=log_interval, tensorboard_log=tensorboard_log, seed=seed
        )

    else:
        # raise NotImplementedError
        for env_name in envs_id:

            if single_model_name is not None:
                # init_wandb_run(project_name, env_name, single_model_name, f"{env_name}/{single_model_name}-{run_tag}")
                # models[single_model_name](env_name, time_steps, log_interval=log_interval,
                #                           tensorboard_log=tensorboard_log, seed=seed)

                weights_path = f"/home/ionelia/weights-benchmark-master/{single_model_name}_{env_name}_{policy_type}"
                if os.path.exists(f"{weights_path}.zip"):
                    models[single_model_name](
                        env_name,
                        time_steps,
                        log_interval=log_interval,
                        tensorboard_log=tensorboard_log,
                        seed=seed,
                        policy=policy_type,
                        load_weights=weights_path
                    )

            else:
                raise NotImplementedError

                runs = api.runs("ionelia/rl-benchmarks").objects

                for run in runs:
                    id_run = run.id
                    name_run = run.name
                    env_name_run, model_name_run = name_run.split("/")

                    if "NOPE" in env_name_run or name_run == 'HalfCheetah-v2/trpo':  # or  "trpo" not in model_name_run:
                        continue
                    else:
                        try:
                            wandb.init(id=id_run, project="rl-benchmarks", resume="must", monitor_gym=True, reinit=True)
                            weights_path = f"/home/ionelia/weights-benchmark-master/{model_name_run}_{env_name_run}_{policy_type}"
                            if os.path.exists(f"{weights_path}.zip"):
                                print(weights_path)
                                models[model_name_run](
                                    env_name_run,
                                    time_steps,
                                    log_interval=log_interval,
                                    tensorboard_log=tensorboard_log,
                                    seed=seed,
                                    policy=policy_type,
                                    load_weights=weights_path
                                )
                        except Exception as e:
                            print(e)

                # for model_name, model_function in models.items():
                # init_wandb_run(project_name, env_name, model_name, f"{env_name}/{model_name}-{run_tag}")
                # model_function(env_name, time_steps, log_interval=log_interval,
                #                tensorboard_log=tensorboard_log, seed=seed)


models = {"ddpg": ddpg, "ppo2": ppo2, "sac": sac, "td3": td3, "trpo": trpo}

if __name__ == "__main__":
    time_steps = int(5000)
    log_interval = 10
    tensorboard_logs = "."  # "/publicwork/students/buzatu/tensorboard-logs"
    args = sys.argv
    run_tag = "2"
    seed = 456
    if len(args) > 1:
        model_name = sys.argv[1]
        if model_name in ["ddpg", "sac", "td3", "trpo", "ppo2"]:

            if len(args) == 3:
                env_name = sys.argv[2]
                run_benchmarks(
                    time_steps=time_steps,
                    single_model_name=model_name,
                    single_env_name=env_name,
                    log_interval=log_interval,
                    tensorboard_log=tensorboard_logs,
                    run_tag=run_tag, seed=seed
                )
            else:
                run_benchmarks(time_steps=time_steps,
                               single_model_name=model_name,
                               log_interval=log_interval,
                               tensorboard_log=tensorboard_logs,
                               run_tag=run_tag, seed=seed)

    else:
        run_benchmarks(time_steps=time_steps,
                       log_interval=log_interval,
                       tensorboard_log=tensorboard_logs,
                       run_tag=run_tag,
                       seed=seed)
