import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        max_episode_steps: int = 400,
    ) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # The TransferCube demonstrations used for training contain 400
        # frames. Override Gymnasium's registered 300-step TimeLimit so the
        # evaluation does not truncate policies before the demonstration
        # horizon. This only changes truncation; the environment's official
        # success condition (reward == 4) remains unchanged.
        self._gym = gymnasium.make(task, obs_type=obs_type, max_episode_steps=max_episode_steps)
        self._task = task

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_seed: int | None = None
        self._episode_results: list[dict[str, int | float | bool | str]] = []

    @override
    def reset(self) -> None:
        self._episode_seed = int(self._rng.integers(2**32 - 1))
        gym_obs, _ = self._gym.reset(seed=self._episode_seed)
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0
        self._episode_steps = 0

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)
        self._episode_steps += 1
        if self._done:
            self._episode_results.append(
                {
                    "task": self._task,
                    "episode_index": len(self._episode_results),
                    "seed": int(self._episode_seed) if self._episode_seed is not None else -1,
                    "success": bool(info.get("is_success", False) or self._episode_reward >= 4),
                    "max_reward": float(self._episode_reward),
                    "steps": self._episode_steps,
                }
            )

    @property
    def episode_results(self) -> tuple[dict[str, int | float | bool | str], ...]:
        return tuple(self._episode_results)

    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
        }
