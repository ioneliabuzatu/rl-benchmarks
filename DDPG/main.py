import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from config import BATCH_SIZE
from config import ENV_NAME
from config import EXPLORATION_NOISE
from config import HIDDEN1
from config import HIDDEN2
from ddpg import DDPG
from replay_buffer import ReplayBuffer


def setup(func):
    def essentials(*args, **kwargs):
        print(args, kwargs)
        torch.manual_seed(100)
        np.random.seed(100)
        func(*args, **kwargs)

    return essentials


@setup
def main(states, actions, hidden1, hidden2, env, max_q_action, tensorboard_dir):
    writer = SummaryWriter(tensorboard_dir)
    agent = DDPG(states, actions, hidden1, hidden2, max_q_action)
    replay_buffer = ReplayBuffer()

    update_parameters_count = 0

    for episode in range(3):
        reward_step = 0
        step = 0
        state = env.reset()

        while True:
            action = agent.select_actions(state)
            action = (action + np.random.normal(0, EXPLORATION_NOISE, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high)

            next_state, reward, done, _ = env.step(action)
            env.render()
            replay_buffer.push((state, next_state, action, reward, np.float(done)))

            state = next_state

            if replay_buffer.__len__() > BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                critic_loss, actor_loss = agent.update_parameters(batch)

                writer.add_scalar("Loss/Critic", critic_loss, global_step=update_parameters_count)
                writer.add_scalar('Loss/Actor', actor_loss, global_step=update_parameters_count)
                writer.add_scalar("Reward", reward_step, global_step=update_parameters_count)

            update_parameters_count += 1

            if done:
                break

            step += 1
            reward_step += reward

    agent.save()


if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    states = env.observation_space.shape[0]
    actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    main(states, actions, HIDDEN1, HIDDEN2, env, max_action, tensorboard_dir="runs")

    env.close()
