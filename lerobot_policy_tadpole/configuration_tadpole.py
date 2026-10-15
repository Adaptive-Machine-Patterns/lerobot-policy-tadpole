# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""Configuration of the Tadpole policy."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig

CAMERAS = ("observation.images.top", "observation.images.side", "observation.images.iso")
# joint angle limits (rad) of the arm; the gripper aperture is 0 (open) .. 1 (closed)
JOINT_LIMITS = (
    (0.1, 2.5132741228718345), (1.5707963267948966, 4.71238898038469),
    (-2.199114857512855, 2.199114857512855), (-1.8849555921538759, 1.8849555921538759),
    (-3.141592653589793, 3.141592653589793), (-3.141592653589793, 3.141592653589793))


@PreTrainedConfig.register_subclass("tadpole")
@dataclass
class TadpoleConfig(PreTrainedConfig):
    """Tadpole: three fixed cameras (top, side, iso, 512 x 512), the 7-number
    state (6 joint angles + gripper) and one instruction clause in; the next
    `chunk_size` states out.

    Images go through a shared CNN (given pixel coordinates as two extra
    channels) to a 16 x 16 grid of tokens per camera; the instruction through
    SmolVLM2's tokenizer and learned word embeddings; the state as one token.
    A transformer reads them with `chunk_size` query tokens and predicts each
    step's change directly (no sampling).

    Training only: each image token must also redraw a small copy of its
    patch (`redraw_weight`), and the state is hidden from `state_dropout` of
    the samples, so the model cannot ignore the cameras.
    """

    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    # all scaling is inside the model, so the processors leave values as they are
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=lambda: {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    })

    image_size: int = 512
    dim: int = 256
    layers: int = 6
    heads: int = 8
    grid: int = 16
    redraw_patch: int = 8
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length: int = 48
    vocab_size: int = 49280
    # The tokenizer ids AMP Tadpole's instructions use, in the order of its
    # word-table rows; any other id reads one extra row ("a word it never
    # learned").  Empty: a row for every id of the tokenizer (vocab_size).
    token_ids: tuple = (2, 260, 272, 284, 288, 543, 614, 826, 1478, 3115, 3154, 3921, 4231, 4870, 5222, 5927, 7842, 9571, 11986, 12536, 13917, 14297, 14362, 14973, 17201, 17967, 19609, 24036, 27865, 28948, 31413, 32577)
    # one step's change is predicted in these units (rad per step x6, gripper per step)
    step_scale: tuple = (0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.125)
    joint_limits: tuple = JOINT_LIMITS

    state_dropout: float = 0.5
    redraw_weight: float = 1.0
    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 1e-10
    optimizer_betas: tuple = (0.9, 0.95)

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if not self.input_features:
            self.input_features = {
                **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, self.image_size, self.image_size))
                   for k in CAMERAS},
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            }
        if not self.output_features:
            self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    def validate_features(self) -> None:
        missing = [k for k in CAMERAS if k not in self.input_features]
        if missing:
            raise ValueError(f"Tadpole needs the cameras {CAMERAS}; missing {missing}")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (7,):
            raise ValueError("Tadpole needs observation.state with 7 numbers (6 joints + gripper)")
        if self.action_feature is None or self.action_feature.shape != (7,):
            raise ValueError("Tadpole predicts 7-number actions (6 joints + gripper)")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, betas=self.optimizer_betas,
                           weight_decay=self.optimizer_weight_decay, grad_clip_norm=10.0)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
