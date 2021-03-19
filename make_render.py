from models import ppo2

env_name = "HumanoidStandup-v2"
policy = "CnnPolicy"

ppo2(env_name, 100, policy=policy)  # , log_interval=log_interval, tensorboard_log=tensorboard_log)
