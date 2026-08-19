import dataclasses
import json
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")
    metrics_path: pathlib.Path = pathlib.Path("data/aloha_sim/metrics.json")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 0
    num_episodes: int = 10
    max_episode_steps: int = 400

    action_horizon: int = 25

    host: str = "0.0.0.0"
    port: int = 8000

    # Set to 0 to run as fast as policy inference and MuJoCo allow. Use 50 to
    # reproduce the wall-clock control frequency of the real ALOHA system.
    max_hz: float = 0.0
    display: bool = False


def main(args: Args) -> None:
    environment = _env.AlohaSimEnvironment(
        task=args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
    )

    runtime.run()

    episodes = list(environment.episode_results)
    success_count = sum(bool(episode["success"]) for episode in episodes)
    summary = {
        "task": args.task,
        "base_seed": args.seed,
        "num_episodes": len(episodes),
        "max_episode_steps": args.max_episode_steps,
        "success_count": success_count,
        "success_rate": success_count / len(episodes) if episodes else 0.0,
        "mean_max_reward": (
            sum(float(episode["max_reward"]) for episode in episodes) / len(episodes) if episodes else 0.0
        ),
        "episodes": episodes,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Evaluation complete: success=%d/%d (%.1f%%), mean max reward=%.3f, metrics=%s",
        success_count,
        len(episodes),
        100.0 * float(summary["success_rate"]),
        float(summary["mean_max_reward"]),
        args.metrics_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
