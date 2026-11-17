"""
DiffusionTrainer — trains a UMI ``DiffusionUnetTimmPolicy``.

This reuses the UMI training stack verbatim: it composes the UMI Hydra
workspace config (from the vendored ``diffusion_policy`` package's
``config/`` dir, bundled inside this plugin), applies overrides derived from the framework's thin
:class:`~solarian.core.configs.TrainConfig`, then instantiates and runs the
workspace exactly like UMI's ``train.py`` does
(``cls = hydra.utils.get_class(cfg._target_); cls(cfg, output_dir).run()``).
The heavy ``accelerate`` multi-GPU loop, EMA, checkpointing and logging all
live in ``TrainDiffusionUnetImageWorkspace`` and are not reimplemented here.

Run standalone / under ``accelerate launch``::

    accelerate launch -m solarian_diffusion.trainer --config cfg.json
"""

import logging
import os
import re
from typing import Optional

from solarian.core.ai.system1.policy_trainer import ModalSpec, PolicyTrainer

logger = logging.getLogger(__name__)


def _parse_dataset_field(dataset_path: str, field: str, default=None):
    """UMI ``parse_dataset_field`` resolver (copied from UMI ``train.py``).

    Some bundled UMI configs reference ``${parse_dataset_field:...}`` to infer
    run metadata from the dataset filename; register it so composing those
    configs doesn't fail.
    """
    name = os.path.basename(str(dataset_path)).split(".")[0]
    tokens = [t for t in name.split("-") if t]

    if field == "data_task":
        return tokens[0] if tokens else default
    if field == "mirror":
        valid = {"black_mirror", "mirror", "swap_mirror", "no_mirror"}
        for t in tokens:
            if t in valid:
                return t
        return default
    if field == "real_data_size":
        for t in tokens:
            m = re.fullmatch(r"real_(\d+)", t)
            if m:
                return int(m.group(1))
        return default
    if field == "sim_data_size":
        for t in tokens:
            m = re.fullmatch(r"sim_(\d+)", t)
            if m:
                return int(m.group(1))
        return default
    if field == "bbox":
        if "bbox" in tokens:
            return "bbox"
        if "no_bbox" in tokens:
            return "no_bbox"
        return default
    raise ValueError(f"Unknown dataset field: {field!r}")


def _register_resolvers() -> None:
    """Register the OmegaConf resolvers the UMI configs rely on."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "parse_dataset_field", _parse_dataset_field, replace=True
    )


def _diffusion_policy_config_dir() -> str:
    """Locate the ``config/`` directory of the vendored ``diffusion_policy`` package."""
    import diffusion_policy

    config_dir = os.path.join(os.path.dirname(diffusion_policy.__file__), "config")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(
            f"Could not find the vendored diffusion_policy config dir at {config_dir}. "
            f"It ships inside the solarian-diffusion plugin; reinstall the "
            f"plugin (pip install -e plugins/solarian-diffusion)."
        )
    return config_dir


class DiffusionTrainer(PolicyTrainer):
    """Trains a UMI diffusion policy by driving the UMI workspace."""

    title = "UMI Diffusion Policy Trainer"

    @classmethod
    def config_class(cls):
        from solarian_diffusion.config import DiffusionTrainConfig

        return DiffusionTrainConfig

    @classmethod
    def modal_spec(cls) -> ModalSpec:
        return ModalSpec(
            python_version="3.10",
            apt_packages=["git", "ffmpeg", "libgl1", "libglib2.0-0"],
            # hydra-core / omegaconf / dill ship with solarian core; here we only
            # add the ML-specific runtimes the vendored diffusion_policy needs.
            pip_packages=[
                "torch",
                "torchvision",
                "accelerate",
                "wandb",
                "diffusers",
                "timm",
                "einops",
                "zarr",
                "numcodecs",
                "imagecodecs",
                "numba",
            ],
            # solarian + this plugin are typically unpublished; ship them as
            # local sources so the container can import them. The vendored
            # `diffusion_policy` / `umi` packages travel with the plugin.
            local_python_packages=[
                "diffusion_policy",
                "umi",
                "solarian",
                "solarian_diffusion",
            ],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dataset_path(self) -> str:
        ds = getattr(self.cfg, "dataset_path", None) or self.cfg.train_dataset
        if isinstance(ds, str) and ds.startswith("local@"):
            name = ds.split("@", 1)[1]
            project = os.getenv("SOLARIAN_PROJECT_PATH", os.getcwd())
            ds = os.path.join(project, "datasets", name)
        return ds

    def _build_overrides(self) -> list:
        cfg = self.cfg
        overrides = [
            f"task.dataset_path={self._resolve_dataset_path()}",
            f"dataloader.batch_size={cfg.batch_size}",
            f"val_dataloader.batch_size={cfg.batch_size}",
            f"dataloader.num_workers={cfg.num_workers}",
            f"training.resume={cfg.resume}",
        ]
        if getattr(cfg, "num_epochs", None) is not None:
            overrides.append(f"training.num_epochs={cfg.num_epochs}")
        logger_cfg = getattr(cfg, "logger", None)
        if logger_cfg is not None and logger_cfg.type != "none":
            backend = logger_cfg.type if isinstance(logger_cfg.type, str) else ",".join(logger_cfg.type)
            overrides.append(f"logging.project={logger_cfg.project}")
            overrides.append(f"++logging.backend='{backend}'")
        # Caller-supplied raw Hydra overrides win (appended last).
        overrides.extend(list(getattr(cfg, "overrides", []) or []))
        return overrides

    def _compose_workspace_cfg(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        _register_resolvers()
        config_dir = _diffusion_policy_config_dir()
        overrides = self._build_overrides()

        # The solarian CLI may already hold an active Hydra context (it composes
        # the project config); clear it so we can compose the UMI config from a
        # different config dir.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=self.cfg.config_name, overrides=overrides)
        return cfg

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> None:
        import hydra

        # If no logger is configured, keep wandb fully offline so accelerate's
        # tracker (used inside the UMI workspace) doesn't try to reach the API.
        if getattr(self.cfg, "logger", None) is None:
            os.environ.setdefault("WANDB_MODE", "offline")

        cfg = self._compose_workspace_cfg()

        logger.info(
            "Composed UMI workspace %s (target=%s, dataset=%s)",
            self.cfg.config_name,
            cfg._target_,
            cfg.task.dataset_path,
        )

        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg, output_dir=self.output_dir)
        workspace.run()


def main():
    DiffusionTrainer.main()


if __name__ == "__main__":
    main()
