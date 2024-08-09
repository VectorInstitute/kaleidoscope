"""
Abstractions for adapting to new backend environments (SLURM, Kubernetes, etc.)

These integrations are stateful and should be concurrency-safe.
"""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

ModelName = str
SLURMJobID = str


class LLMBackendStatus(NamedTuple):
    """Represent status of an LLM Backend."""

    base_url: str | None
    status_text: str | None = None
    raw_status_data: Any | None = None


class LLMBackend(NamedTuple):
    """Represents a vLLM backend running as a SLURM job.

    A job is either not ready (starting up) or ready.
    Jobs that were preempted or stopped should be deleted.

    `is_pending` should be True only if the backend would be ready in
    the near future and does not need to be replaced.

    `slurm_job_id` must be unique.

    """

    model_name: ModelName
    status: LLMBackendStatus

    # Must be unique.
    slurm_job_id: SLURMJobID

    @property
    def is_pending(self) -> bool:
        """Returns whether this backend will be ready in the future."""
        return self.status.status_text in ["PENDING", "LAUNCHING", None]

    @property
    def is_ready(self) -> bool:
        """Returns whether this backend is ready."""
        return self.status.base_url is not None

    @property
    def base_url(self) -> str | None:
        """Alias for base_url."""
        return self.status.base_url


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

    The manager would periodically invoke get_backend_status
    to refresh api base url, and determine if backend was preempted.
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
    def get_backend_status(self, backend: LLMBackend) -> LLMBackendStatus:
        """Return status and API Base URL for backend if ready and not preempted.

        Params:
            backend: LLMBackend to query.

        Returns:
            LLMBackendStatus.

        Design note: not all backend types (e.g., Kubernetes) use SLURM Job ID.
        Request backend instead of job just job_id for greater flexibility.
        """

    @abstractmethod
    def delete_backend(self, backend: LLMBackend) -> None:
        """Request backend deletion.

        This method should not block and should be invoked only once
        for each job_id.
        """
