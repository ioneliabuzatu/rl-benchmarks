import torch
import os
import numpy as np
import random
import wandb
import logging


def init_wandb_run(project_name, env_name, model_name, run_name, dir="/publicwork/students/buzatu/"):
    wandb.init(project=project_name, reinit=True, dir=dir)
    wandb.run.name = f"{env_name}/{model_name}"
    logging.info(f"init run name: {run_name}")


def delete_wandb_run(run_name):
    api = wandb.Api()
    run = api.run(run_name)
    run.delete()
    logging.info(f"run {run_name} had been deleted with success")


def save_model_weights(model, model_name, env_id, policy, path="/publicwork/students/buzatu/weights"):
    model.save(f"{path}/{model_name}_{env_id}_{policy}")


def set_global_seed(seed=123):
    # Tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)


def watch_agent():
    global action, info
    # After training, watch our agent walk
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def example_save_model_properly():
    # Custom MLP policy of three layers of size 128 each
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[128, 128, 128],
                                               feature_extraction="mlp")

    # Create and wrap the environment
    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])

    model = A2C(CustomPolicy, env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=1000)
    model.save("test_save")

    del model

    model = A2C.load("test_save", env=env, policy=CustomPolicy)
