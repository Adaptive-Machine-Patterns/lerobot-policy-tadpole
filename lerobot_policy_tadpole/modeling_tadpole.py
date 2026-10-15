# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The Tadpole network and its LeRobot policy.

Batch keys (after `make_tadpole_pre_post_processors`):

    observation.images.top / .side / .iso   float (B, 3, 512, 512) in [0, 1]
    observation.state                       (B, 7): 6 joint angles (rad) + gripper
    observation.language.tokens             (B, 48) SmolVLM2 token ids of the instruction
    observation.language.attention_mask     (B, 48)
    action                                  (B, chunk, 7), training only: the next states

`predict_action_chunk` returns the next `chunk_size` states (absolute joint
angles + gripper), `select_action` one of them at a time.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_tadpole import CAMERAS, TadpoleConfig

STATE_FEATURES = 31


def state_features(state: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """The 7 state numbers scaled to [-1, 1], plus sines and cosines of the
    joint angles.  (B, 7) -> (B, 31)."""
    q = state[:, :6]
    return torch.cat([2 * (state - lo) / (hi - lo) - 1, torch.sin(2 * q), torch.cos(2 * q),
                      torch.sin(8 * q), torch.cos(8 * q)], dim=1)


class TadpoleNet(nn.Module):
    """images -> shared CNN -> 16 x 16 tokens per camera; words -> embeddings;
    state -> one token; + chunk query tokens -> transformer -> one step per query."""

    def __init__(self, cfg: TadpoleConfig):
        super().__init__()
        d, g, r = cfg.dim, cfg.grid, cfg.redraw_patch
        self.grid = g

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU(),
                nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.GELU())
        self.cnn = nn.ModuleList([                                    # 512 -> 16
            nn.Sequential(nn.Conv2d(5, 48, 7, 4, 3), nn.GroupNorm(8, 48), nn.GELU()),
            block(48, 96), block(96, 192), block(192, d)])
        self.redraw = nn.Linear(d, 3 * r * r)          # token -> its image patch
        self.no_state = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.view = nn.Parameter(torch.randn(len(CAMERAS), 1, d) * 0.02)
        rows = len(cfg.token_ids) + 1 if cfg.token_ids else cfg.vocab_size
        self.word = nn.Embedding(rows, d)
        # tokenizer id -> word-table row (the used ids, then one row for any other)
        if cfg.token_ids:
            row = torch.full((cfg.vocab_size,), len(cfg.token_ids), dtype=torch.long)
            row[torch.tensor(cfg.token_ids)] = torch.arange(len(cfg.token_ids))
        else:
            row = torch.arange(cfg.vocab_size)
        self.register_buffer("token_row", row, persistent=False)
        self.word_pos = nn.Parameter(torch.randn(1, cfg.tokenizer_max_length, d) * 0.02)
        self.state = nn.Linear(STATE_FEATURES, d)
        self.query = nn.Parameter(torch.randn(1, cfg.chunk_size, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 7))

    def forward(self, images, tokens, token_mask, feats, hide_state=None):
        """images (B, 3 cameras, 3, H, W) in [0, 1]; tokens (B, L); token_mask
        (B, L) bool; feats (B, 31); hide_state (B,) bool or None.  Returns the
        per-step changes (B, chunk, 7) and each token's redrawn patch."""
        B, H = images.shape[0], images.shape[-1]
        w = self.word(self.token_row[tokens]) + self.word_pos[:, :tokens.shape[1]]
        ax = torch.linspace(-1, 1, H, device=images.device, dtype=images.dtype)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        f = torch.cat([images.flatten(0, 1) * 2 - 1,
                       xx.expand(B * 3, 1, H, H), yy.expand(B * 3, 1, H, H)], dim=1)
        for stage in self.cnn:
            f = stage(f)
        f = f.flatten(2).transpose(1, 2)
        redrawn = self.redraw(f)
        f = f + self.pos
        f = (f.view(B, 3, self.grid * self.grid, -1) + self.view).flatten(1, 2)
        s = self.state(feats)[:, None]
        if hide_state is not None:
            s = torch.where(hide_state[:, None, None], self.no_state.to(s.dtype), s)
        q = self.query.expand(B, -1, -1)
        x = torch.cat([f, w, s, q], dim=1)
        keep = torch.ones(B, f.shape[1], dtype=torch.bool, device=x.device)
        tail = torch.ones(B, 1 + q.shape[1], dtype=torch.bool, device=x.device)
        pad = ~torch.cat([keep, token_mask, tail], dim=1)
        x = self.fuse(x, src_key_padding_mask=pad)
        return self.head(x[:, -q.shape[1]:]), redrawn


class TadpolePolicy(PreTrainedPolicy):
    config_class = TadpoleConfig
    name = "tadpole"

    def __init__(self, config: TadpoleConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = TadpoleNet(config)
        lim = torch.tensor(config.joint_limits, dtype=torch.float32)
        self.register_buffer("lo", torch.cat([lim[:, 0], torch.zeros(1)]), persistent=False)
        self.register_buffer("hi", torch.cat([lim[:, 1], torch.ones(1)]), persistent=False)
        self.register_buffer("scale", torch.tensor(config.step_scale, dtype=torch.float32),
                             persistent=False)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._queue = deque(maxlen=self.config.n_action_steps)

    def _images(self, batch) -> Tensor:
        x = torch.stack([batch[k] for k in CAMERAS], dim=1).float()     # (B, 3, 3, H, W)
        if x.shape[-1] != self.config.image_size or x.shape[-2] != self.config.image_size:
            B = x.shape[0]
            x = F.interpolate(x.flatten(0, 1), size=(self.config.image_size,) * 2,
                              mode="bilinear", antialias=True, align_corners=False)
            x = x.view(B, 3, 3, self.config.image_size, self.config.image_size)
        return x

    def _run(self, batch, hide=None):
        state = batch[OBS_STATE].float()
        steps, redrawn = self.model(self._images(batch), batch[OBS_LANGUAGE_TOKENS].long(),
                                    batch[OBS_LANGUAGE_ATTENTION_MASK].bool(),
                                    state_features(state, self.lo, self.hi), hide)
        return steps.float(), redrawn.float(), state

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """The next chunk_size states, (B, chunk, 7)."""
        self.eval()
        steps, _, state = self._run(batch)
        return (steps * self.scale).cumsum(dim=1) + state[:, None]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if not self._queue:
            chunk = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Smooth-L1 on the per-step changes, plus the redraw term."""
        B = batch[OBS_STATE].shape[0]
        hide = (torch.rand(B, device=batch[OBS_STATE].device) < self.config.state_dropout
                if self.training and self.config.state_dropout else None)
        steps_pred, redrawn, state = self._run(batch, hide)
        target = batch[ACTION].float()
        steps = torch.diff(torch.cat([state[:, None], target], dim=1), dim=1) / self.scale
        per = F.smooth_l1_loss(steps_pred, steps, reduction="none").mean(-1)       # (B, chunk)
        if "action_is_pad" in batch:
            keep = ~batch["action_is_pad"]
            imitation = (per * keep).sum() / keep.sum().clamp(min=1)
        else:
            imitation = per.mean()
        g, r = self.config.grid, self.config.redraw_patch
        small = F.adaptive_avg_pool2d(self._images(batch).flatten(0, 1), g * r)
        want = F.unfold(small, r, stride=r).transpose(1, 2)
        redraw = ((redrawn - want).abs() * (1 + 20 * want)).mean()
        loss = imitation + self.config.redraw_weight * redraw
        return loss, {"imitation": imitation.item(), "redraw": redraw.item()}
