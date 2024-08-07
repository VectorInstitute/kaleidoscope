"""Unit tests for Backend Launchers"""

import re

import pytest

from web.auto_scaling.backend_launchers import SLURMCLILauncher
from web.auto_scaling.interfaces import AbstractShellCommandExecutor, LLMBackend

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
                return output_template.format(**match.groupdict())

        return self.fallback_output


@pytest.fixture()
def basic_executor() -> DummyShellCommandExecutor:
    """Basic executor that gives output similar to SLURM CLI."""
    templates = {
        r"bash.+launch_server.*": "Example Text\n" "  Submitted batch job 1001 \n",
        r"scontrol show job (1002)": "slurm_load_jobs error: Invalid job id specified",
        r"scontrol show job (1001)": "JobId=1001"
        " JobName=example\nAccount=vector\n"
        " JobState=RUNNING "
        "Example Text\n",
        r"scontrol show job (1010)": "JobId=1010\n JobState=PENDING Example Text\n",
        r"scontrol show job (1011)": "JobId=1011\n JobState=PREEMPTEDExample Text\n",
    }
    return DummyShellCommandExecutor(templates=templates)


def test_launch_model_backend_valid(basic_executor):
    """Test launching a backend given a valid model name."""
    launcher = SLURMCLILauncher(cli_executor=basic_executor)
    backend = launcher.create_backend("Mixtral-8x22B-Instruct-v0.1")
    assert backend.slurm_job_id == "1001"

    assert len(basic_executor.query_history) == 1
    cli_query_str = " ".join(basic_executor.query_history[0])
    assert "--model-family mixtral" in cli_query_str
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
    assert launcher.get_backend_status(LLMBackend("model", EXAMPLE_API_URL, "1001"))
    assert launcher.get_backend_status(LLMBackend("model", None, "1010"))

    # Invalid or otherwise
    assert not launcher.get_backend_status(LLMBackend("model", EXAMPLE_API_URL, "1002"))
    assert not launcher.get_backend_status(LLMBackend("model", EXAMPLE_API_URL, "1011"))


@pytest.mark.parametrize(("base_url",), [(EXAMPLE_API_URL,), (None,)])
def test_delete_model_backend(basic_executor, base_url):
    """Test deleting a model backend."""
    launcher = SLURMCLILauncher(cli_executor=basic_executor)
    launcher.delete_backend(LLMBackend("model", base_url, "1001"))

    assert len(basic_executor.query_history) == 1
    cli_query_str = " ".join(basic_executor.query_history[0])
    assert cli_query_str == "scancel 1001"
