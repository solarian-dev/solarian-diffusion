"""Config schema for the diffusion trainer.

Extends the framework's thin :class:`~solarian.core.configs.TrainConfig` with
the few knobs needed to drive the UMI Hydra workspace.  Everything else
(model architecture, optimizer, scheduler, EMA, ...) lives in the bundled /
``diffusion_policy``-package Hydra config and is overridden via ``overrides``.
"""

from typing import List, Optional

from solarian.core.configs import TrainConfig


class DiffusionTrainConfig(TrainConfig):
    """Validated config for :class:`DiffusionTrainer`."""

    #: Name of the UMI workspace Hydra config to compose (resolved from the
    #: installed ``diffusion_policy`` package's ``config/`` directory).
    config_name: str = "train_diffusion_unet_timm_umi_workspace"

    #: Explicit dataset path (a ``.zarr.zip``). Falls back to ``train_dataset``;
    #: a ``local@<name>`` value resolves under ``$SOLARIAN_PROJECT_PATH/datasets``.
    dataset_path: Optional[str] = None

    #: Number of training epochs (maps to ``training.num_epochs``). UMI trains
    #: by epochs, so this takes precedence over the step-based ``steps`` field.
    num_epochs: Optional[int] = None

    #: Raw Hydra override strings appended last, e.g.
    #: ``["policy.obs_encoder.model_name=vit_base_patch16_clip_224.openai"]``.
    overrides: List[str] = []
