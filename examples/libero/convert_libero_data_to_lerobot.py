"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/tensorflow_datasets_root

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/tensorflow_datasets_root --push_to_hub

To convert each LIBERO subset into a separate output dataset, use:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/tensorflow_datasets_root --separate_subsets

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

from collections.abc import Sequence
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

DEFAULT_REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)  # By default, combine the four public LIBERO suites into one dataset.


def _repo_suffix(raw_dataset_name: str) -> str:
    return raw_dataset_name.removesuffix("_no_noops")


def _convert(
    *,
    data_dir: str,
    repo_name: str,
    raw_dataset_names: Sequence[str],
    push_to_hub: bool,
) -> None:
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in raw_dataset_names:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                observation = step["observation"]
                image = observation["image"] if "image" in observation else observation["image_primary"]
                wrist_image = observation["wrist_image"] if "wrist_image" in observation else observation["image_wrist"]
                language_instruction = step.get("language_instruction", b"")
                task = (
                    language_instruction.decode()
                    if isinstance(language_instruction, bytes | bytearray)
                    else str(language_instruction)
                )
                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": wrist_image,
                        "state": observation["state"],
                        "actions": step["action"],
                        "task": task,
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def main(
    data_dir: str,
    repo_name: str = DEFAULT_REPO_NAME,
    raw_dataset_names: tuple[str, ...] = RAW_DATASET_NAMES,
    *,
    separate_subsets: bool = False,
    push_to_hub: bool = False,
):
    if separate_subsets:
        for raw_dataset_name in raw_dataset_names:
            subset_repo_name = f"{repo_name}_{_repo_suffix(raw_dataset_name)}"
            _convert(
                data_dir=data_dir,
                repo_name=subset_repo_name,
                raw_dataset_names=(raw_dataset_name,),
                push_to_hub=push_to_hub,
            )
    else:
        _convert(
            data_dir=data_dir,
            repo_name=repo_name,
            raw_dataset_names=raw_dataset_names,
            push_to_hub=push_to_hub,
        )


if __name__ == "__main__":
    tyro.cli(main)
