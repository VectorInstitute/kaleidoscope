"""
Abstractions for adapting to new backend environments (SLURM, Kubernetes, etc.)

These integrations are stateful and should be concurrency-safe.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

ModelName = str
SLURMJobID = str


class LLMBackend(NamedTuple):
    """Represents a vLLM backend running as a SLURM job.

    A job is either not ready (starting up) or ready.
    Jobs that were preempted or stopped should be deleted.

    base_url is None denotes that the job is pending.

    slurm_job_id must be unique.
    """

    model_name: ModelName
    base_url: str | None

    # Must be unique.
    slurm_job_id: SLURMJobID

    @property
    def is_ready(self) -> bool:
        """Returns whether this backend is ready."""
        return self.base_url is not None


class AbstractShellCommandExecutor(ABC):
    """Abstraction for running shell commands locally, via SSH, etc.

    These methods should be concurrency safe and will be invoked in parallel."""

    def __init__(self):
        """Optionally, set up stateful environment."""
        return

    @abstractmethod
    def run_shell_command(self, args: list[str]) -> str:
        """Invoke shell command and capture output string."""

    def run_shell_command_detached(self, args: list[str]) -> None:
        """Invoke shell command without capturing output, for performance."""
        self.run_shell_command(args)


class AbstractLLMBackendLauncher(ABC):
    """Abstraction for launching LLM Backends.

    Note that the backend likely will not start immediately.
    `create_backend` should return as soon as a slurm_job_id is obtained, and
    should return a LLMBackend instance where is_ready evaluates to False.

    Backend would inform manager about its status and base_url
    via a callback route. This "launcher" class is only for launching,
    not for retrieving the base_url.
    """

    def __init__(self):
        """Optionally, set up stateful environment."""
        return

    @abstractmethod
    def create_backend(self, model_name: ModelName) -> LLMBackend:
        """Request Backend creation.

        Returns:
            LLMBackend, where slurm_job_id is provided but not base_url.
        """

    @abstractmethod
    def get_backend_status(self, backend: LLMBackend) -> bool:
        """Verify if backend is still available and not preempted.

        Design note: not all backend types (e.g., Kubernetes) use SLURM Job ID.
        Request backend instead of job just job_id for greater flexibility.
        """

    @abstractmethod
    def delete_backend(self, backend: LLMBackend) -> None:
        """Request backend deletion.

        This method should not block and should be invoked only once
        for each job_id.
        """
