"""
Implementations for running shell commands on a remote machine via SSH.
"""

import os
import subprocess
from threading import Semaphore
from typing import NamedTuple

from ..interfaces import AbstractShellCommandExecutor


class _SSHRemoteSpecs(NamedTuple):
    """Configs for a SSH remote machine."""

    username: str | None
    host: str
    port: int = 22

    def get_args_prefix(self) -> list[str]:
        """Return CLI prefix to append to args list.

        If username is None, username will be omitted from command.
        """
        prefix: list[str] = f"ssh -p {self.port}".split(" ")

        if self.username is None:
            prefix += f"{self.host}".split(" ")
        else:
            prefix += f"{self.username}@{self.host}".split(" ")

        return prefix

    @classmethod
    def from_env_var(cls) -> "_SSHRemoteSpecs":
        """Load from credentials specified in environment variables."""

        username = os.environ.get("SSH_USERNAME")
        host = os.environ.get("SSH_HOST", "localhost")
        port = int(os.environ.get("SSH_PORT", 22))

        return _SSHRemoteSpecs(username, host, port)


class RemoteSSHCommandExecutor(AbstractShellCommandExecutor):
    """Run Shell commands on remote machine."""

    def __init__(self):
        self._max_concurrency = int(os.environ.get("MAX_SHELL_CONCURRENCY", 10))
        self._rate_limit_semaphore = Semaphore(self._max_concurrency)

        self._ssh_remote_config = _SSHRemoteSpecs.from_env_var()

    def run_shell_command(self, args: list[str]) -> str:
        """Invoke shell command and capture output string."""
        ssh_args = self._ssh_remote_config.get_args_prefix() + args
        with self._rate_limit_semaphore:
            output = subprocess.run(ssh_args, capture_output=True)
            return output.stdout.decode()

    def run_shell_command_detached(self, args: list[str]) -> None:
        """Invoke shell command without capturing output, for performance."""
        with self._rate_limit_semaphore:
            subprocess.run(args)
