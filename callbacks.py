import gym
import gym.wrappers
from stable_baselines.common.callbacks import BaseCallback
from utils import save_frames_as_gif


def decorator_to_disable_view_window(func):
    def wrapper(*args):
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__

        def constructor(self, *args, **kwargs):
            org_constructor(self, *args, **kwargs)
            self.window.set_visible(visible=False)

        rendering.Viewer.__init__ = constructor
        return func(*args)

    return wrapper


class WandbRenderEnvCallback(BaseCallback):

    def __init__(self, verbose=0, enable_popping_up_window=False, model_name=None, env_name=None):
        super(WandbRenderEnvCallback, self).__init__(verbose)
        self.enable_popping_up_window = enable_popping_up_window
        self.model_name = model_name
        self.env_name = env_name

        self.frames_for_gif = []

    # @decorator_to_disable_view_window
    def record_render_on_wandb(self, before_traning=False):
        if hasattr(self.training_env, "envs"):
            env = gym.wrappers.Monitor(self.training_env.envs[0], "recording", force=True, mode="evaluation")
        else:
            env = gym.wrappers.Monitor(self.training_env, "recording", force=True, mode="evaluation")
        obs = env.reset()
        for i in range(500):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            if dones:
                break

            self.frames_for_gif.append(env.render(mode="rgb_array"))

        env.close()

        if before_traning:
            gif_filename = f"./gifs/{self.env_name}_{self.model_name}_before.gif"
        else:
            gif_filename = f"./gifs/{self.env_name}_{self.model_name}_after.gif"
        save_frames_as_gif(self.frames_for_gif, filename=gif_filename)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # self.record_render_on_wandb(before_traning=True)
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.record_render_on_wandb()
