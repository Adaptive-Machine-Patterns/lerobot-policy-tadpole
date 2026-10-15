# lerobot-policy-tadpole

**[AMP Tadpole](https://huggingface.co/Adaptive-Machine-Patterns/amp-tadpole) as a
[LeRobot](https://github.com/huggingface/lerobot) policy.** AMP Tadpole is a small
vision-language-action model (6.5M parameters) that drives a six-joint arm from three
fixed camera images, the arm's state and a plain-English instruction. It was trained from
scratch on the [AMP Toy Robotics Benchmark](https://github.com/Adaptive-Machine-Patterns/amp-toy-robotics),
where it completes 93.7% of instructions on held-out scenes.

This package adds the policy type `tadpole` to LeRobot, so the model can be loaded,
evaluated and fine-tuned with LeRobot's own tools.

## Install

```bash
pip install git+https://github.com/Adaptive-Machine-Patterns/lerobot-policy-tadpole
```

LeRobot finds the plugin by itself (any installed package named `lerobot_policy_*`).
Needs Python 3.12 and LeRobot 0.6 or later.

## Use

The trained weights are in a separate repository,
[Adaptive-Machine-Patterns/amp-tadpole-lerobot](https://huggingface.co/Adaptive-Machine-Patterns/amp-tadpole-lerobot):

```python
import lerobot_policy_tadpole  # registers the "tadpole" policy type
from lerobot_policy_tadpole import TadpolePolicy, make_tadpole_pre_post_processors

policy = TadpolePolicy.from_pretrained("Adaptive-Machine-Patterns/amp-tadpole-lerobot")
pre, post = make_tadpole_pre_post_processors(policy.config)

chunk = policy.predict_action_chunk(pre({
    "observation.images.top": top,     # (3, H, W) floats in [0, 1]; resized to 512 inside
    "observation.images.side": side,
    "observation.images.iso": iso,
    "observation.state": state,        # 6 joint angles (rad) + gripper (0 open .. 1 closed)
    "task": "Pick up the lime star",
}))                                     # (1, 8, 7): the next 8 joint states + gripper
```

From the command line: `--policy.type=tadpole` for a new model, or
`--policy.path=Adaptive-Machine-Patterns/amp-tadpole-lerobot` for the trained one.

## What it expects

| key | shape | |
|---|---|---|
| `observation.images.top`, `.side`, `.iso` | (3, 512, 512) | the benchmark's three fixed cameras |
| `observation.state` | (7,) | 6 joint angles + gripper opening |
| `task` | text | one instruction clause, in the benchmark's wording |
| `action` (training) | (8, 7) | the next 8 states |

It predicts the 8 steps in one pass (no sampling), as the change at each step, added
up from the current state. For training it adds an auxiliary loss in which each image
token redraws its own patch, and hides the state from half the samples, so that the
model must use its cameras.

**It knows only the benchmark's world:** one arm, three fixed cameras, 8 colours and 5
shapes, and its instruction words. Its word table holds only those words; any other word
reads one shared "never learned" row. It is not suitable for controlling real machinery.

## Licences

- **This code: Apache-2.0** (LICENSE, NOTICE).
- **The weights: CC BY-NC 4.0** (free for non-commercial use), with commercial licences
  from Adaptive Machine Patterns Ltd:
  [adaptive-machine-patterns.com/embodied-ai-workbench](https://adaptive-machine-patterns.com/embodied-ai-workbench/)
  or amp@adaptive-machine-patterns.com.
- Instructions are tokenised with the tokenizer of
  [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
  (Apache-2.0), which LeRobot downloads.

LeRobot and SmolVLM2 are Hugging Face projects; their mention implies no endorsement.
How the code was written: PROVENANCE.md.

## Tests

```bash
pip install -e ".[test]"
pytest
```

© 2026 Adaptive Machine Patterns Ltd, registered in England & Wales, no. 15618869.
