import torch
import numpy as np
import random


def save_model_weights(model,env_id):
    model.save(f"./weights/{model_name}_{env_id}")


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
