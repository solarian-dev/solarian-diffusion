"""
DiffusionPolicyServer — serves a UMI diffusion policy over HTTP.

Wraps the UMI ``DiffusionUnetTimmPolicy`` (loaded from a training checkpoint)
in the generic :class:`~solarian.services.policy.PolicyServer` interface, so
the heavy inference stack (PyTorch + the ``diffusion_policy`` package) lives
on the server side and any skill can drive it over HTTP via
:class:`~solarian.services.policy.PolicyClient`.

Run it standalone (on the robot, the controller, or a dedicated GPU box)::

    solarian run-service diffusion-policy --checkpoint data/models/cup_wild_vit_l_1img.ckpt
    # or
    python -m solarian_diffusion.policy_server -i <ckpt> --port 8001

Install the inference extra to get the heavy deps::

    pip install "solarian-diffusion[serve]"
"""

import logging
import os
from typing import Any, Dict

import numpy as np

from solarian.services import PolicyServer

logger = logging.getLogger(__name__)


class DiffusionPolicyServer(PolicyServer):
    """Policy server hosting a UMI ``DiffusionUnetTimmPolicy``."""

    title = "UMI Diffusion Policy Server"

    def __init__(
        self,
        checkpoint: str,
        num_inference_steps: int = 16,
        fast_init: bool = True,
        device: str | None = None,
    ):
        super().__init__(
            checkpoint=checkpoint,
            num_inference_steps=num_inference_steps,
            fast_init=fast_init,
        )
        self.checkpoint = checkpoint
        self.num_inference_steps = num_inference_steps
        self.fast_init = fast_init
        self._device_arg = device

        self.policy = None
        self.cfg = None
        self.device = None
        self._obs_pose_repr = None
        self._action_pose_repr = None
        self._checkpoint_file = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf, open_dict

        # Imported for its side effect of registering hydra targets, and to
        # type-hint the workspace below.
        from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: F401

        ckpt_path = self.checkpoint
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
        self._checkpoint_file = os.path.basename(ckpt_path)

        logger.info("Loading checkpoint from %s", ckpt_path)
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        self.cfg = cfg
        logger.info("Model: %s", OmegaConf.select(cfg, "policy.obs_encoder.model_name"))

        # Don't fetch pretrained backbone weights — the checkpoint has them.
        try:
            if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is True:
                with open_dict(cfg):
                    cfg.policy.obs_encoder.pretrained = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not disable obs_encoder.pretrained: %s", exc)

        original_use_ema = bool(OmegaConf.select(cfg, "training.use_ema"))
        if self.fast_init and original_use_ema:
            try:
                with open_dict(cfg):
                    cfg.training.use_ema = False
            except Exception as exc:  # noqa: BLE001
                logger.warning("fast_init: could not suppress EMA deepcopy: %s", exc)

        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = workspace_cls(cfg)

        if self.fast_init:
            sds = payload["state_dicts"]
            if original_use_ema and "ema_model" in sds:
                src_key = "ema_model"
            elif "model" in sds:
                src_key = "model"
            else:
                raise RuntimeError(
                    f"Checkpoint has no 'model' or 'ema_model' state_dict; "
                    f"got {list(sds.keys())}"
                )
            logger.info("fast_init: loading '%s' state_dict into model", src_key)
            workspace.model.load_state_dict(sds[src_key], strict=False)
            policy = workspace.model
        else:
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            policy = workspace.model
            if cfg.training.use_ema:
                logger.info("Using EMA model")
                policy = workspace.ema_model

        policy.num_inference_steps = self.num_inference_steps
        self._obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
        self._action_pose_repr = cfg.task.pose_repr.action_pose_repr

        # Device selection: explicit override, else CUDA > MPS > CPU.
        if self._device_arg is not None:
            device = torch.device(self._device_arg)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

        policy.eval().to(device)
        # RTC guidance backpropagates to the action trajectory at every
        # denoising step. Only the trajectory needs a gradient, so dropping it
        # from the weights keeps that graph to the activations alone.
        for param in policy.parameters():
            param.requires_grad_(False)
        self.policy = policy
        logger.info(
            "Policy loaded on %s (obs_pose_repr=%s action_pose_repr=%s steps=%d)",
            device,
            self._obs_pose_repr,
            self._action_pose_repr,
            self.num_inference_steps,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # Real-time chunking masking modes, as named on the wire, mapped to the
    # weight schedules of ``rtc.prefix_weights``. "soft" and "hard" are the two
    # the paper compares; "linear" and "ones" exist for its Figure 8 ablation.
    _RTC_SCHEDULES = {
        "soft": "exp",
        "hard": "zeros",
        "linear": "linear",
        "ones": "ones",
    }

    def predict(
        self,
        obs_dict: Dict[str, np.ndarray],
        prev_action: np.ndarray | None = None,
        inference_delay: int = 0,
        prefix_horizon: int | None = None,
        beta: float | None = None,
        masking: str = "soft",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict one action chunk, optionally inpainted onto ``prev_action``.

        Without ``prev_action`` this is the stock policy path. With it, the
        chunk is generated under real-time chunking: ``prev_action`` is the
        previous chunk already aligned to this chunk's indices (index ``i`` of
        both names the same instant), ``inference_delay`` is how many leading
        actions are already committed, and ``prefix_horizon`` is ``H - s``, one
        past the last index that overlaps the previous chunk.
        """
        import torch
        from diffusion_policy.common.pytorch_util import dict_apply

        from solarian_diffusion import rtc

        device = self.device

        def to_tensor(x):
            tensor = torch.from_numpy(np.asarray(x))
            if device.type == "mps" and tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor.unsqueeze(0).to(device)

        obs_dict_torch = dict_apply(obs_dict, to_tensor)

        if prev_action is None or masking == "off":
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict_torch)
            info = None
        else:
            try:
                schedule = self._RTC_SCHEDULES[masking]
            except KeyError:
                raise ValueError(
                    f"Unknown masking {masking!r}; expected one of "
                    f"{sorted(self._RTC_SCHEDULES)} or 'off'"
                ) from None
            horizon = self.policy.action_horizon
            result = rtc.predict_action(
                self.policy,
                obs_dict_torch,
                prev_action=np.asarray(prev_action, dtype=np.float32),
                inference_delay=int(inference_delay),
                prefix_horizon=horizon if prefix_horizon is None
                else int(prefix_horizon),
                beta=rtc.DEFAULT_BETA if beta is None else float(beta),
                schedule=schedule,
            )
            info = result["info"]

        # (1, H, A) -> (H, A); both action and raw_action are the unnormalized
        # model output. The client applies get_real_umi_action() to convert
        # to executable world poses.
        raw_action = result["action_pred"][0].detach().to("cpu").numpy()
        return {"action": raw_action, "raw_action": raw_action, "info": info}

    def reset(self) -> None:
        import torch

        if self.policy is None:
            return
        self.policy.reset()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        try:
            self.policy.noise_scheduler.set_timesteps(self.policy.num_inference_steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not reset noise scheduler timesteps: %s", exc)
        self.policy.eval()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_info(self) -> Dict[str, Any]:
        from omegaconf import OmegaConf

        return {
            "model_name": OmegaConf.select(self.cfg, "policy.obs_encoder.model_name"),
            "dataset_path": OmegaConf.select(self.cfg, "task.dataset.dataset_path"),
            "obs_pose_repr": self._obs_pose_repr,
            "action_pose_repr": self._action_pose_repr,
            "num_inference_steps": int(self.policy.num_inference_steps),
            "device": str(self.device),
            "checkpoint": self._checkpoint_file or "",
        }

    @property
    def shape_meta(self) -> Dict[str, Any]:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(self.cfg.task.shape_meta, resolve=True)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--input", "-i", required=True,
            help="Path to checkpoint (.ckpt file or a run dir containing checkpoints/latest.ckpt)",
        )
        parser.add_argument(
            "--num-inference-steps", type=int, default=16,
            help="DDIM denoising steps per action chunk (default 16; UMI often fine at 8)",
        )
        parser.add_argument(
            "--no-fast-init", dest="fast_init", action="store_false", default=True,
            help="Disable fast-init (full random init + EMA deepcopy); for debugging",
        )
        parser.add_argument(
            "--device", default=None,
            help="Torch device override (default: cuda > mps > cpu)",
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            checkpoint=args.input,
            num_inference_steps=args.num_inference_steps,
            fast_init=args.fast_init,
            device=args.device,
        )


def main():
    DiffusionPolicyServer.main()


if __name__ == "__main__":
    main()
