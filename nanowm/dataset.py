"""Dataset adapter from PPO Baba samples to NanoWorldModel batches.

NanoWM trains on dictionaries with:
    video:  [T, C, H, W] float tensor, normally normalized to [-1, 1]
    action: [T, action_dim] float tensor
    video_name: stable clip identifier

PPO samples in this project are saved as:
    episode_xxxx/frames.npy   uint8 RGB [N, H, W, 3]
    episode_xxxx/actions.npy  action ids [N - 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset


ResizeMode = Literal["stretch", "pad"]
ActionEncoding = Literal["one_hot", "scalar"]
SliceMode = Literal["exhaustive", "random"]


SMALL_SD_VAE_IMAGE_SIZE = 256
SMALL_SD_VAE_LATENT_SIZE = 32
SMALL_SD_VAE_LATENT_CHANNELS = 4
SMALL_SD_VAE_NUM_FRAMES = 4


@dataclass(frozen=True)
class NanoWMSlice:
    episode_idx: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    num_frames: int
    num_actions: int

    @property
    def valid_frames(self) -> int:
        # PPO stores N frames and N-1 transition actions. Pad one trailing action
        # in the dataset so NanoWM receives action length T for every T frames.
        return min(self.num_frames, self.num_actions + 1)


class BabaSamplesNanoWMDataset(Dataset):
    """Convert PPO `samples/<run>/episode_*` folders to NanoWM-ready clips.

    The default shape matches NanoWM-S/2 with SD-VAE:
        image_size=256 -> SD-VAE latents [4, 32, 32]
        num_frames=4

    Discrete Baba actions are one-hot encoded by default. If your saved actions
    are env action ids 1..4 and you want a compact 4-D action vector, pass
    `action_offset=-1` and `action_dim=4`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        num_frames: int = SMALL_SD_VAE_NUM_FRAMES,
        frame_interval: int = 1,
        image_size: int | Tuple[int, int] = SMALL_SD_VAE_IMAGE_SIZE,
        action_dim: Optional[int] = None,
        action_encoding: ActionEncoding = "one_hot",
        action_offset: int = 0,
        split: Literal["train", "val", "all"] = "all",
        split_ratio: float = 0.9,
        random_seed: int = 42,
        episode_limit: Optional[int] = None,
        source: str = "ppo",
        wm_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        slice_mode: SliceMode = "exhaustive",
        stride: int = 1,
        resize_mode: ResizeMode = "stretch",
        normalize_pixel: bool = True,
    ) -> None:
        super().__init__()
        if frame_interval < 1:
            raise ValueError(f"frame_interval must be >= 1, got {frame_interval}")
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        if slice_mode not in ("exhaustive", "random"):
            raise ValueError(f"slice_mode must be 'exhaustive' or 'random', got {slice_mode!r}")
        if resize_mode not in ("stretch", "pad"):
            raise ValueError(f"resize_mode must be 'stretch' or 'pad', got {resize_mode!r}")
        if action_encoding not in ("one_hot", "scalar"):
            raise ValueError(f"Unsupported action_encoding={action_encoding!r}")

        self.root = Path(root)
        self.num_frames = int(num_frames)
        self.frame_interval = int(frame_interval)
        self.image_size = _as_hw(image_size)
        self.action_encoding = action_encoding
        self.action_offset = int(action_offset)
        self.split = split
        self.split_ratio = float(split_ratio)
        self.random_seed = int(random_seed)
        self.episode_limit = None if episode_limit is None else int(episode_limit)
        self.source = str(source)
        self.wm_loss_weight = float(wm_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        self.slice_mode = slice_mode
        self.stride = int(stride)
        self.resize_mode = resize_mode
        self.normalize_pixel = bool(normalize_pixel)
        self.rng = np.random.RandomState(self.random_seed)

        self.episodes = self._discover_episodes(self.root)
        self.episode_indices = self._split_episode_indices()
        self.episode_indices = self._limit_episode_indices(self.episode_indices)
        self.action_dim = int(action_dim) if action_dim is not None else self._infer_action_dim()
        if self.action_encoding == "scalar":
            self.action_dim = 1

        if self.slice_mode == "exhaustive":
            self.slices = self._build_exhaustive_slices()
        else:
            self.slices = []
            self.valid_episode_indices = [
                idx for idx in self.episode_indices if self._max_start(self.episodes[idx]) >= 0
            ]
            if not self.valid_episode_indices:
                raise ValueError("No episodes are long enough for the requested clip length.")

    def __len__(self) -> int:
        if self.slice_mode == "random":
            return len(self.valid_episode_indices)
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.slice_mode == "random":
            episode_idx = self.valid_episode_indices[idx % len(self.valid_episode_indices)]
            max_start = self._max_start(self.episodes[episode_idx])
            start = int(self.rng.randint(0, max_start + 1)) if max_start > 0 else 0
            end = start + self.num_frames * self.frame_interval
            slice_info = NanoWMSlice(episode_idx, start, end)
        else:
            slice_info = self.slices[idx]

        episode = self.episodes[slice_info.episode_idx]
        frames = np.load(episode.path / "frames.npy", mmap_mode="r")
        actions = np.load(episode.path / "actions.npy", mmap_mode="r")

        frame_indices = self._frame_indices(slice_info.start_frame)
        video = self._frames_to_video(frames[frame_indices])

        action = self._actions_to_tensor(actions, frame_indices)

        video_name = f"{episode.path.name}_start_{slice_info.start_frame:04d}"
        return {
            "video": video,
            "action": action,
            "wm_loss_weight": torch.tensor(self.wm_loss_weight, dtype=torch.float32),
            "action_loss_weight": torch.tensor(self.action_loss_weight, dtype=torch.float32),
            "source": self.source,
            "video_name": video_name,
            "meta_info": {
                "episode_idx": slice_info.episode_idx,
                "episode_path": str(episode.path),
                "start_idx": slice_info.start_frame,
                "source": self.source,
            },
        }

    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        # Compatibility with NanoWM's WorldModelDataset API.
        return {}

    def _discover_episodes(self, root: Path) -> List[EpisodeInfo]:
        if not root.exists():
            raise FileNotFoundError(f"Samples root does not exist: {root}")

        episode_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        episodes: List[EpisodeInfo] = []
        for episode_dir in episode_dirs:
            frames_path = episode_dir / "frames.npy"
            actions_path = episode_dir / "actions.npy"
            if not frames_path.exists() or not actions_path.exists():
                continue

            frames = np.load(frames_path, mmap_mode="r")
            actions = np.load(actions_path, mmap_mode="r")
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"{frames_path} must have shape [N, H, W, 3], got {frames.shape}")
            if actions.ndim not in (1, 2):
                raise ValueError(f"{actions_path} must have shape [N] or [N, D], got {actions.shape}")

            episodes.append(EpisodeInfo(episode_dir, int(frames.shape[0]), int(actions.shape[0])))

        if not episodes:
            raise FileNotFoundError(f"No episode dirs with frames.npy/actions.npy found in {root}")
        return episodes

    def _split_episode_indices(self) -> List[int]:
        all_indices = np.arange(len(self.episodes))
        if self.split == "all":
            return all_indices.tolist()
        if self.split not in ("train", "val"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got {self.split!r}")

        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(all_indices)
        split_at = int(len(all_indices) * self.split_ratio)
        if self.split == "train":
            return all_indices[:split_at].tolist()
        return all_indices[split_at:].tolist()

    def _limit_episode_indices(self, episode_indices: List[int]) -> List[int]:
        if self.episode_limit is None:
            return episode_indices
        if self.episode_limit < 1:
            raise ValueError(f"episode_limit must be >= 1, got {self.episode_limit}")
        if self.episode_limit > len(episode_indices):
            raise ValueError(
                f"Requested episode_limit={self.episode_limit} from {self.root}, "
                f"but only {len(episode_indices)} episodes are available."
            )
        rng = np.random.RandomState(self.random_seed)
        indices = np.asarray(episode_indices)
        rng.shuffle(indices)
        return indices[: self.episode_limit].tolist()

    def _infer_action_dim(self) -> int:
        if self.action_encoding == "scalar":
            return 1
        max_action = None
        for idx in self.episode_indices:
            actions = np.load(self.episodes[idx].path / "actions.npy", mmap_mode="r")
            if actions.ndim != 1:
                raise ValueError("action_dim is required when actions.npy already stores vectors")
            shifted = np.asarray(actions, dtype=np.int64) + self.action_offset
            episode_max = int(shifted.max()) if shifted.size else -1
            max_action = episode_max if max_action is None else max(max_action, episode_max)
        if max_action is None or max_action < 0:
            raise ValueError("Could not infer action_dim from empty actions.")
        return max_action + 1

    def _build_exhaustive_slices(self) -> List[NanoWMSlice]:
        slices: List[NanoWMSlice] = []
        for episode_idx in self.episode_indices:
            episode = self.episodes[episode_idx]
            max_start = self._max_start(episode)
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                end = start + (self.num_frames - 1) * self.frame_interval + 1
                slices.append(NanoWMSlice(episode_idx, start, end))
        if not slices:
            raise ValueError("No valid NanoWM slices; collect longer samples or reduce num_frames/frame_interval.")
        return slices

    def _max_start(self, episode: EpisodeInfo) -> int:
        last_frame_offset = (self.num_frames - 1) * self.frame_interval
        return episode.valid_frames - last_frame_offset - 1

    def _frame_indices(self, start_frame: int) -> np.ndarray:
        return start_frame + np.arange(self.num_frames) * self.frame_interval

    def _frames_to_video(self, frames: np.ndarray) -> torch.Tensor:
        video = torch.as_tensor(np.asarray(frames), dtype=torch.float32).permute(0, 3, 1, 2)
        video = video / 255.0
        if video.shape[-2:] != self.image_size:
            video = _resize_video(video, self.image_size, self.resize_mode)
        if self.normalize_pixel:
            video = video * 2.0 - 1.0
        return video.contiguous()

    def _actions_to_tensor(self, actions: np.ndarray, frame_indices: np.ndarray) -> torch.Tensor:
        action_indices = frame_indices[:-1]
        if actions.ndim == 2:
            padded = torch.zeros((self.num_frames, actions.shape[1]), dtype=torch.float32)
            valid = action_indices < actions.shape[0]
            if np.any(valid):
                rows = torch.as_tensor(np.nonzero(valid)[0], dtype=torch.long)
                padded[rows] = torch.as_tensor(
                    np.asarray(actions[action_indices[valid]]),
                    dtype=torch.float32,
                )
            return padded

        shifted = np.asarray(actions, dtype=np.int64) + self.action_offset
        if self.action_encoding == "scalar":
            out = torch.zeros((self.num_frames, 1), dtype=torch.float32)
            valid = action_indices < shifted.shape[0]
            if np.any(valid):
                rows = torch.as_tensor(np.nonzero(valid)[0], dtype=torch.long)
                out[rows, 0] = torch.as_tensor(
                    shifted[action_indices[valid]],
                    dtype=torch.float32,
                )
            return out

        out = torch.zeros((self.num_frames, self.action_dim), dtype=torch.float32)
        valid = action_indices < shifted.shape[0]
        if np.any(valid):
            values = torch.as_tensor(shifted[action_indices[valid]], dtype=torch.long)
            if torch.any(values < 0) or torch.any(values >= self.action_dim):
                raise ValueError(
                    f"Action ids must fit [0, {self.action_dim}); got min={int(values.min())}, max={int(values.max())}"
                )
            out[torch.as_tensor(np.nonzero(valid)[0], dtype=torch.long), values] = 1.0
        return out


def create_train_val_datasets(
    root: str | Path,
    **kwargs: Any,
) -> Tuple[BabaSamplesNanoWMDataset, BabaSamplesNanoWMDataset]:
    """Create train/val datasets with the same episode split settings."""
    train = BabaSamplesNanoWMDataset(root, split="train", **kwargs)
    val = BabaSamplesNanoWMDataset(root, split="val", **kwargs)
    return train, val


def create_replaced_train_dataset(
    ppo_root: str | Path,
    random_root: str | Path,
    *,
    random_fraction: float,
    total_episodes: Optional[int] = None,
    ppo_action_loss_weight: float = 1.0,
    random_action_loss_weight: float = 0.0,
    wm_loss_weight: float = 1.0,
    **kwargs: Any,
) -> ConcatDataset:
    """Create a train dataset where a fraction of PPO episodes is replaced by random episodes."""
    if not 0.0 <= float(random_fraction) <= 1.0:
        raise ValueError(f"random_fraction must be in [0, 1], got {random_fraction}")

    ppo_count = count_sample_episodes(ppo_root)
    random_count = count_sample_episodes(random_root)
    if total_episodes is None:
        total_episodes = ppo_count
    total_episodes = int(total_episodes)
    if total_episodes < 1:
        raise ValueError(f"total_episodes must be >= 1, got {total_episodes}")

    random_episodes = int(round(total_episodes * float(random_fraction)))
    ppo_episodes = total_episodes - random_episodes
    if ppo_episodes > ppo_count:
        raise ValueError(f"Need {ppo_episodes} PPO episodes, but {ppo_root} has {ppo_count}.")
    if random_episodes > random_count:
        raise ValueError(f"Need {random_episodes} random episodes, but {random_root} has {random_count}.")

    datasets = []
    if ppo_episodes > 0:
        datasets.append(
            BabaSamplesNanoWMDataset(
                ppo_root,
                split="all",
                episode_limit=ppo_episodes,
                source="ppo",
                wm_loss_weight=wm_loss_weight,
                action_loss_weight=ppo_action_loss_weight,
                **kwargs,
            )
        )
    if random_episodes > 0:
        datasets.append(
            BabaSamplesNanoWMDataset(
                random_root,
                split="all",
                episode_limit=random_episodes,
                source="random",
                wm_loss_weight=wm_loss_weight,
                action_loss_weight=random_action_loss_weight,
                **kwargs,
            )
        )
    if not datasets:
        raise ValueError("Mixed train dataset is empty.")
    return ConcatDataset(datasets)


def count_sample_episodes(root: str | Path) -> int:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Samples root does not exist: {root}")
    return sum(
        1
        for episode_dir in root.iterdir()
        if episode_dir.is_dir()
        and (episode_dir / "frames.npy").exists()
        and (episode_dir / "actions.npy").exists()
    )


def _as_hw(image_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) == 1:
        return int(image_size[0]), int(image_size[0])
    if len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"image_size must be int or length-1/2 sequence, got {image_size!r}")


def _resize_video(video: torch.Tensor, image_size: Tuple[int, int], resize_mode: ResizeMode) -> torch.Tensor:
    if resize_mode == "stretch":
        return F.interpolate(video, size=image_size, mode="bilinear", align_corners=False)

    target_h, target_w = image_size
    _, _, height, width = video.shape
    scale = min(target_h / height, target_w / width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return F.pad(resized, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
