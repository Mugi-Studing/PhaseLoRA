# ruff: noqa: E402

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Literal


def _configure_mujoco_gl_backend() -> None:
    """Prefer a headless backend when X11 is unavailable."""
    current_backend = os.environ.get("MUJOCO_GL")
    display = os.environ.get("DISPLAY")

    if current_backend and current_backend.lower() != "glx":
        return

    if display and _x11_display_looks_available(display):
        if current_backend is None:
            os.environ["MUJOCO_GL"] = "glx"
        return

    if current_backend != "osmesa":
        if display:
            logging.warning(
                "DISPLAY %s is unavailable; switching MUJOCO_GL to osmesa for headless rendering.",
                display,
            )
        os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.pop("DISPLAY", None)


def _x11_display_looks_available(display: str) -> bool:
    """Best-effort check for a local X11 socket backing DISPLAY."""
    if not display.startswith(":"):
        return True

    display_num = display[1:].split(".", maxsplit=1)[0]
    if not display_num.isdigit():
        return True

    x11_probe = None
    if shutil.which("xset"):
        x11_probe = ["xset", "-display", display, "q"]
    elif shutil.which("xdpyinfo"):
        x11_probe = ["xdpyinfo", "-display", display]

    if x11_probe is not None:
        try:
            result = subprocess.run(
                x11_probe,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except Exception:
            pass
        else:
            return result.returncode == 0

    return pathlib.Path(f"/tmp/.X11-unix/X{display_num}").exists()


_configure_mujoco_gl_backend()

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    request_timeout_s: float | None = 180.0
    connect_timeout_s: float | None = 30.0
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 5  # Number of rollouts per task
    # Start evaluating from this task index (0-based). Useful for resuming after interruption.
    start_task_id: int = 0
    # Optional inclusive end task index (0-based). If None, evaluate until suite end.
    end_task_id: int | None = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True  # Whether to write mp4 replay videos
    continue_on_video_write_error: bool = True  # Do not abort eval if ffmpeg/imageio video write fails
    results_out_path: str | None = None  # Optional JSON results output

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    classification = _load_task_classification(args.task_suite_name)
    task_stats: list[dict[str, Any]] = []

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_host = args.host
    if client_host in {"0.0.0.0", "::"}:
        logging.warning("Client host %s is non-routable; using 127.0.0.1 for local connection.", client_host)
        client_host = "127.0.0.1"

    benchmark_root = get_libero_path("benchmark_root")
    logging.info("Active LIBERO benchmark_root: %s", benchmark_root)

    client = _websocket_client_policy.WebsocketClientPolicy(
        client_host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        recv_timeout_s=args.request_timeout_s,
    )
    server_metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", server_metadata)

    # Start evaluation
    task_start = max(0, int(args.start_task_id))
    if args.end_task_id is None:
        task_stop_exclusive = num_tasks_in_suite
    else:
        task_stop_exclusive = min(num_tasks_in_suite, int(args.end_task_id) + 1)
    if task_start >= task_stop_exclusive:
        raise ValueError(
            f"Invalid task range: start_task_id={task_start}, end_task_id={args.end_task_id}, num_tasks={num_tasks_in_suite}"
        )

    logging.info("Evaluating task ids in [%d, %d]", task_start, task_stop_exclusive - 1)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(task_start, task_stop_exclusive)):
        # Get task
        task = task_suite.get_task(task_id)
        task_name = task.name
        category = None
        difficulty = None
        if classification is not None:
            entry = classification.get(task_name)
            if entry is not None:
                category = entry.get("category")
                difficulty = entry.get("difficulty_level")

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        num_trials = min(args.num_trials_per_task, len(initial_states))
        for episode_idx in tqdm.tqdm(range(num_trials)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            client.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img, wrist_img = _extract_images(obs, args.resize_size, flip_mode="rot180")

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": _build_state(obs, state_format="axis_angle"),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode.
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                try:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                except Exception as e:
                    if args.continue_on_video_write_error:
                        logging.error("Video write failed for task_id=%d episode=%d: %s", task_id, episode_idx, e)
                    else:
                        raise

            # Log current results
            logging.info(f"Success: {done}")
            _log_running_metrics(
                total_episodes=total_episodes,
                total_successes=total_successes,
                task_episodes=task_episodes,
                task_successes=task_successes,
                category=category,
                difficulty=difficulty,
            )

        # Log final results
        _log_task_summary(
            task_name=task_name,
            task_description=task_description,
            task_episodes=task_episodes,
            task_successes=task_successes,
            total_episodes=total_episodes,
            total_successes=total_successes,
        )

        task_stats.append(
            {
                "suite": args.task_suite_name,
                "task_id": task_id,
                "task_name": task_name,
                "language": task_description,
                "category": category,
                "difficulty_level": difficulty,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": (float(task_successes) / float(task_episodes)) if task_episodes else 0.0,
            }
        )

        if args.results_out_path:
            _write_results_snapshot(
                args=args,
                task_stats=task_stats,
                total_episodes=total_episodes,
                total_successes=total_successes,
                classification=classification,
                server_metadata=server_metadata,
                task_start_id=task_start,
                task_end_id_inclusive=task_stop_exclusive - 1,
                num_tasks_in_suite=num_tasks_in_suite,
            )

    total_success_rate = (float(total_successes) / float(total_episodes)) if total_episodes else 0.0
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    category_stats: dict[str, dict[str, Any]] = {}
    if classification is not None:
        category_stats = _summarize_category_stats(task_stats)
        for category, stats in sorted(category_stats.items()):
            logging.info(
                "Category %s success: %d/%d (%.1f%%)",
                category,
                stats["successes"],
                stats["episodes"],
                stats["success_rate"] * 100.0,
            )

    if args.results_out_path:
        _write_results_snapshot(
            args=args,
            task_stats=task_stats,
            total_episodes=total_episodes,
            total_successes=total_successes,
            classification=classification,
            server_metadata=server_metadata,
            task_start_id=task_start,
            task_end_id_inclusive=task_stop_exclusive - 1,
            num_tasks_in_suite=num_tasks_in_suite,
        )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _load_task_classification(suite_name: str) -> dict[str, dict[str, Any]] | None:
    classification_path = pathlib.Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    if not classification_path.exists():
        logging.info("Task classification file not found at %s; category stats disabled.", classification_path)
        return None
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)
    suite_entries = classification.get(suite_name)
    if not suite_entries:
        logging.info("No classification entries for suite %s; category stats disabled.", suite_name)
        return None
    return {entry["name"]: entry for entry in suite_entries}


def _summarize_category_stats(task_stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for stats in task_stats:
        category = stats.get("category") or "Unknown"
        successes = int(stats.get("successes", 0))
        episodes = int(stats.get("episodes", 0))
        entry = counts.setdefault(category, {"successes": 0, "episodes": 0})
        entry["successes"] += successes
        entry["episodes"] += episodes

    summary: dict[str, dict[str, Any]] = {}
    for category, entry in counts.items():
        episodes = entry["episodes"]
        success_rate = (entry["successes"] / episodes) if episodes else 0.0
        summary[category] = {
            "successes": entry["successes"],
            "episodes": episodes,
            "success_rate": success_rate,
        }
    return summary


def _write_results_snapshot(
    *,
    args: Args,
    task_stats: list[dict[str, Any]],
    total_episodes: int,
    total_successes: int,
    classification: dict[str, dict[str, Any]] | None,
    server_metadata: dict[str, Any] | None,
    task_start_id: int,
    task_end_id_inclusive: int,
    num_tasks_in_suite: int,
) -> None:
    output_path = pathlib.Path(args.results_out_path) if args.results_out_path else None
    if output_path is None:
        return

    total_success_rate = (float(total_successes) / float(total_episodes)) if total_episodes else 0.0
    category_stats: dict[str, dict[str, Any]] = {}
    if classification is not None:
        category_stats = _summarize_category_stats(task_stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "suite": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "task_start_id": task_start_id,
        "task_end_id_inclusive": task_end_id_inclusive,
        "num_tasks_in_suite": num_tasks_in_suite,
        "num_tasks_completed": len(task_stats),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "category_stats": category_stats,
        "server_metadata": server_metadata,
        "task_stats": task_stats,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    logging.info("Wrote results snapshot to %s", output_path)


def _log_running_metrics(
    *,
    total_episodes: int,
    total_successes: int,
    task_episodes: int,
    task_successes: int,
    category: str | None,
    difficulty: str | None,
) -> None:
    total_success_rate = (float(total_successes) / float(total_episodes)) if total_episodes else 0.0
    task_success_rate = (float(task_successes) / float(task_episodes)) if task_episodes else 0.0
    category_suffix = f", category={category}" if category else ""
    difficulty_suffix = f", difficulty={difficulty}" if difficulty else ""
    message = (
        f"Running metrics: episodes={total_episodes}, successes={total_successes}, "
        f"success_rate={total_success_rate * 100.0:.1f}% | "
        f"task_success_rate={task_success_rate * 100.0:.1f}%"
        f"{category_suffix}{difficulty_suffix}"
    )
    logging.info(message)
    tqdm.tqdm.write(message, file=sys.stdout)


def _log_task_summary(
    *,
    task_name: str,
    task_description: str,
    task_episodes: int,
    task_successes: int,
    total_episodes: int,
    total_successes: int,
) -> None:
    task_success_rate = (float(task_successes) / float(task_episodes)) if task_episodes else 0.0
    total_success_rate = (float(total_successes) / float(total_episodes)) if total_episodes else 0.0
    message = (
        f"Task {task_name} ({task_description}) summary: "
        f"{task_successes}/{task_episodes} success ({task_success_rate * 100.0:.1f}%); "
        f"total={total_successes}/{total_episodes} ({total_success_rate * 100.0:.1f}%)"
    )
    logging.info(message)
    tqdm.tqdm.write(message, file=sys.stdout)


def _extract_images(
    obs: dict,
    resize_size: int,
    *,
    flip_mode: Literal["rot180", "flipud", "none"] = "rot180",
) -> tuple[np.ndarray, np.ndarray]:
    img = np.ascontiguousarray(obs["agentview_image"])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    if flip_mode == "rot180":
        img = np.ascontiguousarray(img[::-1, ::-1])
        wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1])
    elif flip_mode == "flipud":
        img = np.ascontiguousarray(np.flipud(img))
        wrist_img = np.ascontiguousarray(np.flipud(wrist_img))
    elif flip_mode != "none":
        raise ValueError(f"Unknown flip mode: {flip_mode}")
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def _build_state(obs: dict, *, state_format: Literal["quat", "axis_angle"]) -> np.ndarray:
    if state_format == "quat":
        return np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"][:1]))
    if state_format == "axis_angle":
        return np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )
    raise ValueError(f"Unknown state format: {state_format}")


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero)
