# SPDX-FileCopyrightText: 2026 Adaptive Machine Patterns Ltd
# SPDX-License-Identifier: Apache-2.0
"""Tadpole as a LeRobot policy plugin.

LeRobot imports any installed package whose name starts with
`lerobot_policy_`; importing this one registers the policy type "tadpole"
(`--policy.type=tadpole`), its model and its processors.

    configuration_tadpole.py   TadpoleConfig
    modeling_tadpole.py        TadpolePolicy (and the network, TadpoleNet)
    processor_tadpole.py       make_tadpole_pre_post_processors
"""

from .configuration_tadpole import TadpoleConfig
from .modeling_tadpole import TadpolePolicy
from .processor_tadpole import make_tadpole_pre_post_processors

__all__ = ["TadpoleConfig", "TadpolePolicy", "make_tadpole_pre_post_processors"]
