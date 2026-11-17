"""Real-time chunking (RTC) for the UMI DDIM diffusion policy.

Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow
Policies* (NeurIPS 2025) — https://www.pi.website/research/real_time_chunking

RTC generates the next action chunk while the previous one is still executing.
The actions that will unavoidably have been consumed by the time inference
finishes are "frozen" to the values the controller is already committed to, and
the rest is inpainted around them, so consecutive chunks cannot bifurcate into
different strategies at the seam.

The paper writes its guidance term for flow matching, but this checkpoint is
DDIM with epsilon prediction. Rather than swap the tuned sampler for a flow one,
the corrections below are mapped into the DDIM parameterisation; see
``guidance_weight`` and ``guided_sample`` for the two places the map shows up.
Every formula is cross-checked against the authors' reference implementation,
https://github.com/Physical-Intelligence/real-time-chunking-kinetix (``src/model.py``).
"""

import math
from typing import Any, Dict, Literal

import numpy as np

# The paper's max_guidance_weight, from Table 4 and the ablation in Figure 7.
DEFAULT_BETA = 5.0

PrefixSchedule = Literal["exp", "linear", "ones", "zeros"]


def prefix_weights(
    total: int,
    start: int,
    end: int,
    schedule: PrefixSchedule = "exp",
) -> np.ndarray:
    """Soft mask over chunk indices (paper Eq. 5).

    ``start`` is the inference delay ``d``: up to there the previous chunk is
    already committed, so those actions are attended to at full weight.
    ``end`` is ``H - s``, one past the last index that overlaps the previous
    chunk; from there on there is nothing to agree with and the weight is 0.
    In between the weight decays, because an action further into the future was
    predicted from a staler observation and deserves less trust.

    With ``start=2, end=6, total=10`` the "linear" schedule gives
    ``1 1 4/5 3/5 2/5 1/5 0 0 0 0``. "exp" is the paper's default; "zeros" is
    hard masking (attend to the committed prefix only) and "ones" attends to
    the whole overlap at full weight. Both are kept for the Figure 8 ablation.

    ``end`` wins over ``start``: if ``end < start`` then ``start`` is pushed
    down to it, so ``end == 0`` ignores the previous chunk entirely.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    start = min(start, end)
    index = np.arange(total, dtype=np.float64)
    if schedule == "ones":
        w = np.ones(total, dtype=np.float64)
    elif schedule == "zeros":
        w = (index < start).astype(np.float64)
    elif schedule in ("linear", "exp"):
        w = np.clip((start - 1 - index) / (end - start + 1) + 1, 0.0, 1.0)
        if schedule == "exp":
            # Normalised by expm1(1) rather than the algebraically equal e - 1,
            # which lands a ULP the wrong side of it and leaves the committed
            # prefix at 1.0000000000000002.
            w = w * np.expm1(w) / np.expm1(1.0)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return np.where(index >= end, 0.0, w)


def guidance_weight(alpha_bar: float, beta: float = DEFAULT_BETA) -> float:
    """The paper's ΠGDM guidance weight, in DDIM's noise schedule.

    In flow time the weight is ``min(beta, (1 - t) / (t * r_t^2))`` with
    ``r_t^2 = (1 - t)^2 / (t^2 + (1 - t)^2)``, which simplifies to
    ``(t^2 + (1 - t)^2) / (t * (1 - t))``. Substituting the standard
    flow/variance-preserving correspondence ``t = sqrt(a) / (sqrt(a) + sqrt(1 - a))``
    for ``a = alpha_bar`` collapses the numerator to ``a + (1 - a) = 1``:

        weight = 1 / (sqrt(a) * sqrt(1 - a))

    Same U shape as Figure 7 and the same minimum of 2 at ``alpha_bar = 0.5``,
    diverging at both ends of the schedule — which is why the clip at ``beta``
    is not optional. The paper's value of 5 transfers unchanged, because the
    substitution is an identity rather than a reparameterisation.
    """
    denom = math.sqrt(max(alpha_bar, 0.0)) * math.sqrt(max(1.0 - alpha_bar, 0.0))
    if denom <= 0.0:
        return float(beta)
    return float(min(beta, 1.0 / denom))


def guided_sample(
    policy,
    global_cond,
    target,
    weights,
    beta: float = DEFAULT_BETA,
    generator=None,
):
    """DDIM sampling with the previous chunk inpainted via ΠGDM guidance.

    ``target`` and the returned trajectory are in the policy's normalized
    action space; ``weights`` is the per-timestep mask of shape ``(H,)``.

    Mirrors ``DiffusionUnetTimmPolicy.conditional_sample`` step for step, with
    one addition: before each ``scheduler.step`` the denoised estimate is nudged
    toward ``target`` along the pseudoinverse-guidance direction.
    """
    import torch

    model = policy.model
    # Kept as the checkpoint was tuned, clip_sample and all. The clip bounds
    # how hard the prefix can be pinned — it truncates the correction once the
    # denoised estimate reaches +-1 — but every legitimate target is a real
    # action and so already inside that range, and sampling without it lets the
    # freshly generated tail leave the range the normalizer was fitted to,
    # which is an out-of-distribution pose on a real arm.
    scheduler = policy.noise_scheduler
    prediction_type = scheduler.config.prediction_type
    if prediction_type not in ("epsilon", "sample"):
        raise NotImplementedError(
            f"RTC guidance supports epsilon and sample prediction, "
            f"not {prediction_type!r}"
        )

    trajectory = torch.randn(
        size=target.shape,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )

    scheduler.set_timesteps(policy.num_inference_steps)
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=target.device, dtype=target.dtype
    )
    # (H,) -> (1, H, 1): one weight per chunk index, shared by every action dim,
    # which is how the paper applies diag(W).
    mask = weights.reshape(1, -1, 1)

    for t in scheduler.timesteps:
        alpha_bar = alphas_cumprod[t]
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).sqrt()

        # Grad is needed only through this one forward pass, and only with
        # respect to the trajectory; the observation encoder ran once, outside.
        with torch.enable_grad():
            x = trajectory.detach().requires_grad_(True)
            model_output = model(x, t, local_cond=None, global_cond=global_cond)
            if prediction_type == "epsilon":
                denoised = (x - sqrt_one_minus * model_output) / sqrt_alpha
            else:
                denoised = model_output
            error = ((target - denoised) * mask).detach()
            # sum(denoised * error) differentiates to J^T error, the
            # vector-Jacobian product of Eq. 2.
            (correction,) = torch.autograd.grad((denoised * error).sum(), x)

        weight = guidance_weight(float(alpha_bar), beta)
        # The paper corrects the velocity field, which moves the denoised
        # estimate by (1 - t) * weight * g_flow. The flow latent is the DDIM
        # latent divided by sqrt(a) + sqrt(1 - a), so g_flow picks up that same
        # factor by the chain rule and (1 - t) * (sqrt(a) + sqrt(1 - a))
        # collapses to sqrt(1 - a). Pushing that through
        # denoised = (x - sqrt(1 - a) * eps) / sqrt(a) cancels the sqrt(1 - a),
        # which is what keeps this stable at the low-noise end where it is tiny.
        if prediction_type == "epsilon":
            model_output = model_output.detach() - sqrt_alpha * weight * correction
        else:
            model_output = (
                model_output.detach() + sqrt_one_minus * weight * correction
            )

        trajectory = scheduler.step(
            model_output,
            t,
            trajectory.detach(),
            generator=generator,
            **policy.kwargs,
        ).prev_sample

    return trajectory


def predict_action(
    policy,
    obs_dict: Dict[str, Any],
    prev_action,
    inference_delay: int,
    prefix_horizon: int,
    beta: float = DEFAULT_BETA,
    schedule: PrefixSchedule = "exp",
) -> Dict[str, Any]:
    """``DiffusionUnetTimmPolicy.predict_action``, with RTC guidance.

    ``prev_action`` is the previous chunk already aligned to this chunk's
    indices — index ``i`` of both names the same instant — in unnormalized
    action units, shape ``(B, H, A)``. ``prefix_horizon`` is ``H - s``.

    Returns the same ``action`` / ``action_pred`` keys as the stock path, plus
    the weights actually used and how closely the frozen prefix was matched.
    """
    import torch

    assert "past_action" not in obs_dict, "not implemented yet"

    horizon = policy.action_horizon
    if prefix_horizon < 0 or prefix_horizon > horizon:
        raise ValueError(
            f"prefix_horizon must be in [0, {horizon}], got {prefix_horizon}"
        )
    if inference_delay < 0 or inference_delay > horizon:
        raise ValueError(
            f"inference_delay must be in [0, {horizon}], got {inference_delay}"
        )

    nobs = policy.normalizer.normalize(obs_dict)
    batch = next(iter(nobs.values())).shape[0]
    with torch.no_grad():
        global_cond = policy.obs_encoder(nobs)

    target = torch.as_tensor(
        np.asarray(prev_action), device=policy.device, dtype=policy.dtype
    )
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.shape != (batch, horizon, policy.action_dim):
        raise ValueError(
            f"prev_action must be {(batch, horizon, policy.action_dim)}, "
            f"got {tuple(target.shape)}"
        )
    ntarget = policy.normalizer["action"].normalize(target)

    weights = prefix_weights(horizon, inference_delay, prefix_horizon, schedule)
    weights_t = torch.as_tensor(weights, device=policy.device, dtype=policy.dtype)

    nsample = guided_sample(
        policy,
        global_cond=global_cond,
        target=ntarget,
        weights=weights_t,
        beta=beta,
    )
    action_pred = policy.normalizer["action"].unnormalize(nsample)

    # How far the committed prefix moved despite being pinned. Small means the
    # guidance bound; large means it did not, and the seam will still jump. The
    # worst single action dimension, so it mixes the units the action space
    # mixes and is a diagnostic rather than a distance.
    frozen = weights >= 1.0
    if frozen.any():
        residual = (
            (action_pred[:, frozen] - target[:, frozen]).abs().max().item()
        )
    else:
        residual = 0.0

    return {
        "action": action_pred,
        "action_pred": action_pred,
        "info": {
            "rtc_weights": weights.tolist(),
            "rtc_frozen_residual": residual,
            "rtc_inference_delay": int(inference_delay),
            "rtc_prefix_horizon": int(prefix_horizon),
        },
    }
