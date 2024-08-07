"""
Implementations for running shell commands on local machine.
"""

import os
import subprocess
from threading import Semaphore

from ..interfaces import AbstractShellCommandExecutor


class LocalShellCommandExecutor(AbstractShellCommandExecutor):
    """Run Shell commands on remote machine."""

    def __init__(self):
        self._max_concurrency = int(os.environ.get("MAX_SHELL_CONCURRENCY", 10))
        self._rate_limit_semaphore = Semaphore(self._max_concurrency)

    def run_shell_command(self, args: list[str]) -> str:
        """Invoke shell command and capture output string."""
        with self._rate_limit_semaphore:
            output = subprocess.run(args, capture_output=True)
            return output.stdout.decode()

    def run_shell_command_detached(self, args: list[str]) -> None:
        """Invoke shell command without capturing output, for performance."""
        with self._rate_limit_semaphore:
            subprocess.run(args)
