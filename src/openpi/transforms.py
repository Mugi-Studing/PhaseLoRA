from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import pathlib
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class InjectOfflineCoarseFineLabel(DataTransformFn):
    """Injects precomputed coarse/fine labels using the sample index.

    The label file can be either:
    - JSON: {"labels_by_index": {"<index>": <numeric_label_or_score>, ...}}
    - JSON: {"labels_by_index": {"<index>": {"P": <score>, "E": <score>}, ...}}
    - JSON: {"<index>": <numeric_label_or_score>, ...}
    - NPZ: arrays named "index" and "label"
    """

    label_path: str
    index_key: str = "index"
    output_key: str = "route_label"
    primary_component_key: str = "P"
    secondary_component_key: str = "E"
    history_component_key: str = "router_action_history"
    history_component_fallback_key: str = "H"
    strict: bool = True
    only_when_actions_present: bool = True
    _labels_by_index: dict[int, float | dict[str, Any]] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        path = pathlib.Path(self.label_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Offline coarse/fine label file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "labels_by_index" in payload:
                payload = payload["labels_by_index"]
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSON label payload in {path}: expected dict")

            labels_by_index: dict[int, float | dict[str, float]] = {}
            for k, v in payload.items():
                idx = int(k)
                if isinstance(v, dict):
                    parsed: dict[str, float | list] = {}
                    for comp_k, comp_v in v.items():
                        comp_key = str(comp_k)
                        if isinstance(comp_v, int | float):
                            parsed[comp_key] = float(comp_v)
                        else:
                            parsed[comp_key] = comp_v
                    labels_by_index[idx] = parsed
                else:
                    labels_by_index[idx] = float(v)
        elif suffix == ".npz":
            data = np.load(path)
            if "index" not in data or "label" not in data:
                raise ValueError(f"Invalid NPZ label payload in {path}: expected arrays 'index' and 'label'")
            indices = np.asarray(data["index"]).reshape(-1)
            labels = np.asarray(data["label"]).reshape(-1)
            labels_by_index = {
                int(index): float(label)
                for index, label in zip(indices.tolist(), labels.tolist(), strict=False)
            }
        else:
            raise ValueError(f"Unsupported label file type for {path}; expected .json or .npz")

        object.__setattr__(self, "_labels_by_index", labels_by_index)

    def __call__(self, data: DataDict) -> DataDict:
        if self.only_when_actions_present and "actions" not in data:
            return data

        if self.index_key not in data:
            if self.strict:
                raise ValueError(f"Cannot inject route label without '{self.index_key}' in sample")
            return data

        sample_index = int(np.asarray(data[self.index_key]).item())
        label = self._labels_by_index.get(sample_index)
        if label is None:
            if self.strict:
                raise KeyError(f"No offline route label found for sample index {sample_index}")
            label = 0.0

        if isinstance(label, dict):
            out = {**data}

            primary_value = None
            if self.primary_component_key in label:
                primary_value = label[self.primary_component_key]
            elif label:
                primary_value = next(iter(label.values()))
            else:
                primary_value = 0.0

            out[self.output_key] = np.asarray(primary_value, dtype=np.float32)

            if self.primary_component_key in label:
                out["route_label_p"] = np.asarray(label[self.primary_component_key], dtype=np.float32)
            if self.secondary_component_key in label:
                out["route_label_e"] = np.asarray(label[self.secondary_component_key], dtype=np.float32)
            bimanual_components = {
                "P_l": "route_label_p_left",
                "E_l": "route_label_e_left",
                "P_r": "route_label_p_right",
                "E_r": "route_label_e_right",
                "C": "route_label_coordination",
            }
            for component_key, output_key in bimanual_components.items():
                if component_key in label:
                    out[output_key] = np.asarray(label[component_key], dtype=np.float32)

            history_value = None
            if self.history_component_key in label:
                history_value = label[self.history_component_key]
            elif self.history_component_fallback_key in label:
                history_value = label[self.history_component_fallback_key]
            if history_value is not None:
                out["router_action_history"] = np.asarray(history_value, dtype=np.float32)
            return out

        return {
            **data,
            self.output_key: np.asarray(label, dtype=np.float32),
        }


@dataclasses.dataclass(frozen=True)
class ConditionDropout(DataTransformFn):
    prob: float = 0.0
    drop_prompt: bool = True
    drop_image: bool = True
    drop_state: bool = True
    only_when_actions_present: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        if self.prob <= 0.0:
            return data
        if self.only_when_actions_present and "actions" not in data:
            return data
        if np.random.rand() >= self.prob:
            return data

        if self.drop_prompt and "prompt" in data:
            data["prompt"] = np.asarray("")

        if self.drop_image and "image" in data:
            data["image"] = {k: np.zeros_like(v) for k, v in data["image"].items()}
            if "image_mask" in data:
                data["image_mask"] = {k: np.zeros((), dtype=np.bool_) for k in data["image_mask"]}

        if self.drop_state and "state" in data:
            data["state"] = np.zeros_like(data["state"])

        return data


@dataclasses.dataclass(frozen=True)
class FrequencyBlurImages(DataTransformFn):
    """Applies stochastic frequency-domain low-pass filtering to image inputs.

    This transform keeps low-frequency structure (coarse shape / color blocks) while
    progressively releasing high-frequency detail as sampled t increases.
    """

    prob: float = 0.0
    min_cutoff_ratio: float = 0.08
    max_cutoff_ratio: float = 1.0
    only_when_actions_present: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        if self.prob <= 0.0:
            return data
        if "image" not in data:
            return data
        if self.only_when_actions_present and "actions" not in data:
            return data
        if np.random.rand() >= self.prob:
            return data

        t = self._extract_t(data)
        min_ratio = float(np.clip(self.min_cutoff_ratio, 0.0, 1.0))
        max_ratio = float(np.clip(self.max_cutoff_ratio, min_ratio, 1.0))
        cutoff_ratio = min_ratio + (max_ratio - min_ratio) * t

        data["image"] = {k: self._lowpass_image(v, cutoff_ratio) for k, v in data["image"].items()}
        return data

    def _lowpass_image(self, image: np.ndarray, cutoff_ratio: float) -> np.ndarray:
        x = np.asarray(image)
        if x.ndim != 3:
            return x

        h, w = x.shape[0], x.shape[1]
        radius_limit = 0.5 * min(h, w)
        cutoff_radius = max(1.0, cutoff_ratio * radius_limit)

        yy, xx = np.ogrid[:h, :w]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = radius <= cutoff_radius

        x_float = x.astype(np.float32)
        filtered = np.empty_like(x_float)
        for c in range(x.shape[2]):
            freq = np.fft.fftshift(np.fft.fft2(x_float[..., c]))
            freq *= mask
            filtered[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(freq)))

        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            filtered = np.clip(filtered, info.min, info.max)
            return filtered.astype(x.dtype)
        return filtered.astype(x.dtype, copy=False)

    def _extract_t(self, data: DataDict) -> float:
        # Prefer externally provided diffusion timestep if available.
        for key in ("t", "timestep", "diffusion_t", "diffusion_timestep"):
            if key in data:
                value = np.asarray(data[key])
                if value.size == 1:
                    return float(np.clip(value.item(), 0.0, 1.0))
        return float(np.random.rand())


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        out = apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

        # Keep router history in the same normalized action domain as model actions.
        # Offline labels store raw executed actions, while inference history is collected
        # from model outputs before output unnormalization.
        if "router_action_history" in out:
            flat_stats = flatten_dict(self.norm_stats)
            action_stats = None
            if "actions" in flat_stats:
                action_stats = flat_stats["actions"]
            elif "action" in flat_stats:
                action_stats = flat_stats["action"]

            if action_stats is not None:
                history = np.asarray(out["router_action_history"])
                normalized_history = (
                    self._normalize_quantile(history, action_stats)
                    if self.use_quantiles
                    else self._normalize(history, action_stats)
                )
                out = {**out, "router_action_history": normalized_history.astype(np.float32, copy=False)}

        return out

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
