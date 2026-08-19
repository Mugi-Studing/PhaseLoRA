import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Policy server parameters.
    host: str = "127.0.0.1"
    port: int = 8000
    request_timeout_s: float | None = 180.0
    connect_timeout_s: float | None = 30.0
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO task selection.
    task_suite_name: str = "libero_10"
    task_id: int = 0
    init_state_id: int = 0
    env_seed: int = 7
    num_steps_wait: int = 10

    # Noise sweep settings.
    num_rollouts: int = 8
    noise_seed: int = 0
    reuse_same_noise_every_replan: bool = True
    action_horizon: int = 10
    action_dim: int = 32

    # Output.
    video_out_path: str = "data/libero/noise_sweep_videos"


def eval_fixed_seed_noise_sweep(args: Args) -> None:
    np.random.seed(args.env_seed)
    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not (0 <= args.task_id < task_suite.n_tasks):
        raise ValueError(
            f"task_id {args.task_id} out of range for {args.task_suite_name} with {task_suite.n_tasks} tasks"
        )

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not (0 <= args.init_state_id < len(initial_states)):
        raise ValueError(
            f"init_state_id {args.init_state_id} out of range for task {args.task_id} with {len(initial_states)} initial states"
        )

    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.env_seed)
    max_steps = _get_max_steps(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        recv_timeout_s=args.request_timeout_s,
    )
    server_metadata = client.get_server_metadata()

    task_segment = task_description.replace(" ", "_")
    summary = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "init_state_id": args.init_state_id,
        "env_seed": args.env_seed,
        "noise_seed": args.noise_seed,
        "num_rollouts": args.num_rollouts,
        "reuse_same_noise_every_replan": args.reuse_same_noise_every_replan,
        "server_metadata": server_metadata,
        "results": [],
    }

    logging.info("Evaluating task %s (%s), fixed init_state_id=%d", args.task_id, task_description, args.init_state_id)

    for rollout_idx in range(args.num_rollouts):
        env.reset()
        client.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[args.init_state_id])
        replay_images = []
        t = 0
        done = False
        replan_idx = 0

        rollout_noise_rng = np.random.default_rng(args.noise_seed + rollout_idx)
        noise_shape = _infer_noise_shape(server_metadata, args)
        fixed_noise = rollout_noise_rng.standard_normal(noise_shape, dtype=np.float32)

        logging.info(
            "Starting rollout %d/%d with noise seed %d and noise shape %s",
            rollout_idx + 1,
            args.num_rollouts,
            args.noise_seed + rollout_idx,
            noise_shape,
        )

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            replay_images.append(img)

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                if args.reuse_same_noise_every_replan:
                    noise = fixed_noise
                else:
                    step_rng = np.random.default_rng(args.noise_seed + rollout_idx * 10_000 + replan_idx)
                    noise = step_rng.standard_normal(noise_shape, dtype=np.float32)

                action_chunk = client.infer(element, noise=noise)["actions"]
                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
                action_plan.extend(action_chunk[: args.replan_steps])
                replan_idx += 1

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if done:
                break

        suffix = "success" if done else "failure"
        rollout_stub = f"task{args.task_id:02d}_{task_segment}_init{args.init_state_id:02d}_rollout{rollout_idx:02d}"
        video_path = out_dir / f"{rollout_stub}_{suffix}.mp4"
        imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

        noise_path = out_dir / f"{rollout_stub}_noise.npy"
        np.save(noise_path, fixed_noise)

        result = {
            "rollout_idx": rollout_idx,
            "success": bool(done),
            "steps": t,
            "video_path": str(video_path),
            "noise_path": str(noise_path),
            "noise_seed": args.noise_seed + rollout_idx,
            "num_replans": replan_idx,
        }
        summary["results"].append(result)
        logging.info("Rollout %d finished: success=%s, steps=%d, video=%s", rollout_idx, done, t, video_path)

    summary_path = out_dir / f"task{args.task_id:02d}_{task_segment}_init{args.init_state_id:02d}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved summary to %s", summary_path)


def _infer_noise_shape(server_metadata: dict, args: Args) -> tuple[int, int]:
    action_horizon = server_metadata.get("action_horizon", args.action_horizon)
    action_dim = server_metadata.get("action_dim", args.action_dim)
    return int(action_horizon), int(action_dim)


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.array(quat, copy=True)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_fixed_seed_noise_sweep)
