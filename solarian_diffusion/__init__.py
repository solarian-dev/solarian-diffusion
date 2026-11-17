"""solarian-diffusion — UMI diffusion policy plugin.

A *policy plugin* (keyed by model family, not by skill) that owns both halves
of a UMI ``DiffusionUnetTimmPolicy``:

* :class:`DiffusionTrainer` — a :class:`solarian.core.ai.system1.PolicyTrainer`
  that trains the policy by reusing the UMI ``diffusion_policy`` workspace
  (the accelerate multi-GPU loop). Discovered by ``solarian train`` via the
  ``solarian.trainers`` entry point (``trainer: diffusion``).
* :class:`DiffusionPolicyServer` — a :class:`solarian.services.PolicyServer`
  that serves a trained checkpoint over HTTP, advertised as the
  ``diffusion-policy`` service (``solarian run-service diffusion-policy``).

Skills (e.g. ``solarian-skill-cup-arrange``) consume the served policy over
HTTP via ``PolicyClient`` and stay independent of this plugin.
"""

__all__ = ["DiffusionTrainer", "DiffusionPolicyServer"]


def __getattr__(name):
    # Lazy attribute access so importing the package (e.g. for entry-point
    # discovery) doesn't drag in torch / diffusion_policy unless actually used.
    if name == "DiffusionTrainer":
        from solarian_diffusion.trainer import DiffusionTrainer

        return DiffusionTrainer
    if name == "DiffusionPolicyServer":
        from solarian_diffusion.policy_server import DiffusionPolicyServer

        return DiffusionPolicyServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
