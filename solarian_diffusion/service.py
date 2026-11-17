from solarian.services import Service


class DiffusionPolicyService(Service):
    """A trained UMI diffusion checkpoint, served by :class:`DiffusionPolicyServer`."""

    server_module = "solarian_diffusion.policy_server"
    default_port = 8001
    checkpoint_arg = "--input"

    @classmethod
    def modal_spec(cls):
        # Served in the environment its checkpoints were trained in.
        from solarian_diffusion.trainer import DiffusionTrainer

        return DiffusionTrainer.modal_spec()
