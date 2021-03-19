import os
from pathlib import Path
# from utils import models_names_dict
import wandb

def main():
    run_id = input("paste run id:")
    wandb.init(id=run_id, project="rl-benchmarks", resume="must")
    # weights_path = "/home/ionelia/weights-benchmark-master"
    # files = [weights_model_name for weights_model_name in os.listdir(weights_path) if
    #          weights_model_name.endswith("zip")]
    # for file in files:
    #     weights_file_name_without_extention = Path(file).stem
    #     model_name, env_name, polity_type = weights_file_name_without_extention.split("_")
    #
    #     model = models_names_dict[model_weights_name_wothout_extention].load("ddpg_mountain")
    #
    #     obs = env.reset()
    #     while True:
    #         action, _states = model.predict(obs)
    #         obs, rewards, dones, info = env.step(action)
    #         env.render()


if __name__ == '__main__':
    main()
