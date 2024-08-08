"""Unit tests for Backend Launchers"""

import re

import pytest

from web.auto_scaling.backend_launchers import SLURMCLILauncher
from web.auto_scaling.interfaces import (
    AbstractShellCommandExecutor,
    LLMBackend,
    LLMBackendStatus,
)

EXAMPLE_API_URL = "http://localhost:8001/"


class DummyShellCommandExecutor(AbstractShellCommandExecutor):
    """CLI Executor for unit testing."""

    def __init__(
        self,
        templates: dict[str, str] | None = None,
        fallback_output: str = "",
    ):
        """
        Params:
            templates: map regexp pattern for input to output.
                Use named capture groups for substitution in output template.
                e.g., {r"scontrol show job (?P<job_id>[0-9]+)": "Job {job_id}"}
        """
        if templates is None:
            templates = {}
        self.templates = templates

        self.fallback_output = fallback_output

        self.query_history: list[list[str]] = []

    def run_shell_command(self, args: list[str]) -> str:
        """Log executed shell command and possibly return matched output."""

        self.query_history.append(args)
        command = " ".join(args)
        for pattern, output_template in self.templates.items():
            match = re.match(pattern, command, re.DOTALL)
            if match is not None:
                return output_template

        return self.fallback_output


@pytest.fixture()
def basic_executor() -> DummyShellCommandExecutor:
    """Basic executor that gives output similar to SLURM CLI."""
    templates = {
        r".*launch.*": ('\n\n {"slurm_job_id": "1001", "base_url":  "UNAVAILABLE"} \n'),
        r".*status --json-mode 1002$": (
            '\n{"model_status": "LOG_FILE_NOT_FOUND", "base_url": "UNAVAILABLE"}'
        ),
        r".*status --json-mode 1001$": (
            '\n{"model_status": "READY", "base_url": "' + EXAMPLE_API_URL + '"}'
        ),
        r".*status --json-mode 1010$": (
            '\n{"model_status": "READY", "base_url": "' + EXAMPLE_API_URL + '"}'
        ),
        r".*status --json-mode 1011$": (
            '\n{"model_status": "PENDING", "base_url": "UNAVAILABLE"}'
        ),
    }
    return DummyShellCommandExecutor(templates=templates)


def test_launch_model_backend_valid(basic_executor):
    """Test launching a backend given a valid model name."""
    launcher = SLURMCLILauncher(cli_executor=basic_executor)
    backend = launcher.create_backend("Mixtral-8x22B-Instruct-v0.1")
    assert backend.slurm_job_id == "1001"

    assert len(basic_executor.query_history) == 1
    cli_query_str = " ".join(basic_executor.query_history[0])
    assert "mixtral" in cli_query_str
    assert "--model-variant 8x22B-Instruct-v0.1" in cli_query_str


@pytest.mark.parametrize(
    ("invalid_model_name",),
    [
        ("Mixture-8x22B-Instruct-v0.1",),
        ("Mixtral-8x22B-Instruct-v0.1; whoami",),
        ("Mixtral-8x22B-Instruct-v0.1 ; whoami",),
        ("Mixtral-8x22B-Instruct-v0.1 && whoami",),
        ("Mixtral-8x22B-Instruct-v0.1 | cat | whoami",),
    ],
)
def test_launch_model_backend_invalid(basic_executor, invalid_model_name):
    """Test launching a backend given an invalid model name.

    Include shell escape validation tests here.
    """
    launcher = SLURMCLILauncher(cli_executor=basic_executor)

    with pytest.raises(ValueError):
        launcher.create_backend(invalid_model_name)


def test_get_model_backend_status(basic_executor):
    """Test retrieving model backend status ."""
    launcher = SLURMCLILauncher(cli_executor=basic_executor)

    # READY or PENDING
    for job_id in ["1001", "1010"]:
        assert (
            launcher.get_backend_status(
                LLMBackend("model", LLMBackendStatus(None), True, job_id)
            ).base_url
            == EXAMPLE_API_URL
        )

    # Invalid or otherwise
    assert (
        launcher.get_backend_status(
            LLMBackend("model", LLMBackendStatus(EXAMPLE_API_URL), False, "1002")
        ).base_url
        is None
    )
    assert (
        launcher.get_backend_status(
            LLMBackend("model", LLMBackendStatus(EXAMPLE_API_URL), False, "1011")
        ).base_url
        is None
    )


@pytest.mark.parametrize(("base_url",), [(EXAMPLE_API_URL,), (None,)])
def test_delete_model_backend(basic_executor, base_url):
    """Test deleting a model backend."""
    launcher = SLURMCLILauncher(cli_executor=basic_executor)
    launcher.delete_backend(
        LLMBackend("model", LLMBackendStatus(base_url), False, "1001")
    )

    assert len(basic_executor.query_history) == 1
    cli_query_str = " ".join(basic_executor.query_history[0])
    assert cli_query_str.split(" ")[-2:] == ["shutdown", "1001"]
