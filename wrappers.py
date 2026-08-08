import gymnasium as gym
import ale_py

def make_env(env_id: str = "ALE/Galaxian-v5", render_mode: str | None = None) -> gym.Env:
    gym.register_envs(ale_py)

    env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1
    )
    return env

def preProcess(env: gym.Env) -> gym.Env:
    env = gym.wrappers.AtariPreprocessing(
        env,
        grayscale_obs=True,
        screen_size=84,
        frame_skip=4,
        terminal_on_life_loss=False
    )

    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env