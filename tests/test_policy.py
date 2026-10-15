# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""The plugin registers with LeRobot, trains, and, with the published
weights, answers exactly as AMP Tadpole's own model package does.

The weight tests read AMP_TADPOLE_LEROBOT and AMP_TADPOLE (local folders),
or download the two Hugging Face repos; they are skipped offline."""

import importlib.util
import os

import numpy as np
import pytest
import torch

from lerobot_policy_tadpole import TadpoleConfig, TadpolePolicy, make_tadpole_pre_post_processors

CAMS = ("top", "side", "iso")


def frame(images, state, clause):
    f = {f"observation.images.{v}": torch.from_numpy(images[i]).float() / 255 for i, v in enumerate(CAMS)}
    f["observation.state"] = torch.tensor(state, dtype=torch.float32)
    f["task"] = clause
    return f


def folder(env, repo):
    if os.environ.get(env):
        return os.environ[env]
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{repo} not available: {e}")


def test_registered_with_lerobot():
    from lerobot.policies.factory import get_policy_class
    assert get_policy_class("tadpole") is TadpolePolicy


def test_word_rows():
    """Only the instructions' tokens get rows, plus one for any other word."""
    p = TadpolePolicy(TadpoleConfig(device="cpu"))
    n = len(p.config.token_ids)
    assert p.model.word.num_embeddings == n + 1
    ids = torch.tensor([p.config.token_ids[3], 7, p.config.token_ids[0]])
    assert p.model.token_row[ids].tolist() == [3, n, 0]


def test_training_step_and_action_queue():
    torch.manual_seed(0)
    cfg = TadpoleConfig(device="cpu")
    pol = TadpolePolicy(cfg)
    batch = {f"observation.images.{v}": torch.rand(2, 3, 64, 64) for v in CAMS}     # resized inside
    batch["observation.state"] = torch.zeros(2, 7)
    batch["observation.language.tokens"] = torch.tensor([list(cfg.token_ids[:4]) + [2] * 44] * 2)
    batch["observation.language.attention_mask"] = torch.tensor([[1] * 4 + [0] * 44] * 2)
    batch["action"] = torch.zeros(2, cfg.chunk_size, 7)
    pol.train()
    loss, info = pol.forward(batch)
    loss.backward()
    assert torch.isfinite(loss) and {"imitation", "redraw"} <= set(info)
    pol.eval()
    pol.reset()
    acts = [pol.select_action(batch) for _ in range(cfg.n_action_steps)]
    assert all(a.shape == (2, 7) for a in acts) and not pol._queue


def test_same_answers_as_the_model_package():
    cameras = pytest.importorskip("amp_toy_robotics.cameras")
    from amp_toy_robotics import core
    from safetensors.torch import load_file

    pol = TadpolePolicy.from_pretrained(folder("AMP_TADPOLE_LEROBOT", "Adaptive-Machine-Patterns/amp-tadpole-lerobot"))
    pol.eval()
    pkg_dir = folder("AMP_TADPOLE", "Adaptive-Machine-Patterns/amp-tadpole")
    spec = importlib.util.spec_from_file_location("amp_tadpole_model", os.path.join(pkg_dir, "model.py"))
    pkg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pkg)
    ref = pkg.build()
    ref.load_state_dict(load_file(os.path.join(pkg_dir, "weights.safetensors")))
    ref.eval()
    pre, _ = make_tadpole_pre_post_processors(pol.config)
    for seed in (9_000_301, 9_000_302):
        scene = core.new_scene(seed)
        images = cameras.observation(scene.q, scene.objects, 0.0)
        state = np.append(scene.q, 0.0)
        clause = f"Pick up the {scene.object_names[0]}"
        want = pkg.predict(ref, images, state, clause)
        got = pol.predict_action_chunk(pre(frame(images, state, clause)))[0].numpy()
        assert np.abs(got - want).max() < 1e-4
