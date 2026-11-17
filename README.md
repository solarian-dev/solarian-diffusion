# solarian-diffusion

A **policy plugin** for the Solarian framework that trains and serves a
[UMI](https://umi-gripper.github.io/) `DiffusionUnetTimmPolicy`.

Unlike a *skill* plugin (a behavior), a *policy* plugin is keyed to a model
family/library and ships both halves of a policy:

- **`DiffusionTrainer`** — a `solarian.core.ai.system1.PolicyTrainer` that
  trains the policy by reusing the UMI `diffusion_policy` workspace (the
  `accelerate` multi-GPU loop). Discovered by `solarian train` through the
  `solarian.trainers` entry point (`trainer: diffusion`).
- **`DiffusionPolicyServer`** — a `solarian.core.ai.system1.PolicyServer` that
  serves a trained checkpoint over HTTP.

The UMI model + training code (the `diffusion_policy` package — timm workspace,
`UmiDataset`, model code, and the bundled Hydra configs — plus the small
`umi.common.pose_util` helper) is **vendored directly into this plugin**, so
there is no separate UMI install to manage. The HTTP-serving stack,
`hydra-core`, `omegaconf` and `dill` ship with `solarian` core; only the ML
runtimes (PyTorch, diffusers, timm, …) are optional extras.

## Install

```bash
# Trainer + server discovery only (no torch):
pip install solarian-diffusion

# Training (accelerate + the diffusion model stack):
pip install "solarian-diffusion[train]"

# Serving a checkpoint:
pip install "solarian-diffusion[serve]"
```

## Train

Add a training file, `conf/training/<name>.yaml`, and set `trainer: diffusion`:

```yaml
trainer: diffusion
job_name: cup_diffusion
train_dataset: /data/datasets/cup_in_the_wild.zarr.zip   # or local@cup_in_the_wild.zarr.zip
batch_size: 64
num_workers: 4
num_epochs: 120
config_name: train_diffusion_unet_timm_umi_workspace
overrides: []        # raw Hydra overrides, applied last
logger:
  type: wandb
  project: umi
```

```bash
solarian train --conf conf/training/<name>
```

The trainer composes the named UMI workspace config from the vendored
`diffusion_policy` package, applies the overrides above, and runs
`workspace.run()` — the same loop as UMI's `train.py`.

### Cloud (Modal)

Install `solarian-modal` and run `solarian modal-train --conf conf/training/<name>` with
a `modal:` block (`gpu_type`, `gpus`, `volume`, ...) in the config. It builds
the image from this plugin's `modal_spec()`, uploads the local dataset the
config names, and runs training remotely with `accelerate launch`.

## Serve

```bash
solarian-diffusion-policy --input <checkpoint.ckpt> --port 8001
```

Point any skill (e.g. `solarian-skill-cup-arrange`) at it via
`policy_server_url`.

## License

[Apache License 2.0](https://github.com/solarian-dev/solarian-diffusion/blob/main/LICENSE).
Copyright 2026 PrimaryKey Inc. The vendored `diffusion_policy` and `umi`
packages are derived from
[Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface)
and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (MIT);
see [NOTICE](https://github.com/solarian-dev/solarian-diffusion/blob/main/NOTICE).

Solarian and the Solarian logo are trademarks of PrimaryKey Inc. See the
[trademark policy](https://github.com/solarian-dev/solarian/blob/main/TRADEMARKS.md).
