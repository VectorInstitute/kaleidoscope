"""
Logic for launching backend via SLURM CLI.

Note that exact logic for running CLI command is specified
in the shell `executor` implementation.
"""

import logging
import re
from pathlib import Path
from typing import NamedTuple

from ..executors import LocalShellCommandExecutor
from ..interfaces import (
    AbstractLLMBackendLauncher,
    AbstractShellCommandExecutor,
    LLMBackend,
    ModelName,
)

# Important: filter out unsafe characters from user-specified "variant".
# Map regexp for model name to Vector Inference "model family", capturing
# Vector Inference "model variant"
MODEL_FAMILIES: dict[str, str] = {
    r"^Mixtral-(?P<variant>[a-zA-Z0-9\.\-]+)$": "mixtral",
    r"^Mistral-(?P<variant>[a-zA-Z0-9\-]+)$": "mistral",
    r"^(Meta-)?Llama-3-(?P<variant>[a-zA-Z0-9\-]+)$": "llama3",
    r"^(Meta-)?Llama-2-(?P<variant>[a-zA-Z0-9\-]+)$": "llama2",
    r"^c4ai-command-r-(?P<variant>[a-zA-Z0-9\-]+)$": "command-r",
}
SLURM_JOB_SUBMIT_PATTERN = r".*Submitted batch job (?P<slurm_job_id>\d+)([\s\n].*)?"
SLURM_JOB_STATUS_PATTERN = r".*JobState=(?P<status>\w+)\s.+"
VALID_SLURM_STATUSES = ["PENDING", "RUNNING"]


class VectorInferenceModelConfig(NamedTuple):
    """
    Configs for invoking Vector Inference.
    """

    model_family: str
    model_variant: str

    @classmethod
    def from_model_name(cls, model_name: str) -> "VectorInferenceModelConfig | None":
        """Parse model name using heuristics.

        Params:
            model_name: e.g., gemma-2b-it

        Returns:
            _VectorInferenceModelConfig or None, if not parsed
        """
        for pattern_str, model_family in MODEL_FAMILIES.items():
            match = re.match(pattern_str, model_name)
            if match is not None:
                model_variant = match.groupdict().get("variant", "")
                return VectorInferenceModelConfig(model_family, model_variant)


class SLURMCLILauncher(AbstractLLMBackendLauncher):
    """
    Launch LLM Backend via SLURM CLI.

    Note that the backend likely will not start immediately.
    `create_backend` should return as soon as a slurm_job_id is obtained, and
    should return a LLMBackend instance where is_ready evaluates to False.

    Backend would inform manager about its status and base_url
    via a callback route. This "launcher" class is only for launching,
    not for retrieving the base_url.
    """

    def __init__(
        self,
        cli_executor: AbstractShellCommandExecutor | None = None,
        vector_inference_base_path: str | None = None,
    ):
        """
        Optionally, provide cli_executor instance.

        Otherwise, a default one would be selected.
        """
        self.logger = logging.getLogger(__name__)

        if cli_executor is None:
            cli_executor = LocalShellCommandExecutor()
        self.cli_executor = cli_executor

        if vector_inference_base_path is None:
            current_dir = Path(__file__).resolve()
            self.logger.info(f"current_dir: {current_dir}")
            self.vector_inference_base_path = Path.joinpath(
                current_dir.parents[3],
                "vendors",
                "vector_inference",
            )
        else:
            self.vector_inference_base_path = Path(vector_inference_base_path)

    def create_backend(self, model_name: ModelName) -> LLMBackend:
        """Request Backend creation.

        Returns:
            LLMBackend, where slurm_job_id is provided but not base_url.
        """
        model_config = VectorInferenceModelConfig.from_model_name(model_name)
        if model_config is None:
            raise ValueError(f"{model_name} is not known.")

        launch_shell_script_path = Path.joinpath(
            self.vector_inference_base_path,
            "src",
            "launch_server.sh",
        )

        # must validate and filter out shell escapes.
        launch_command = (
            f"bash {launch_shell_script_path}"
            f" --model-family {model_config.model_family}"
            f" --model-variant {model_config.model_variant}"
        )
        launch_args = launch_command.split(" ")

        self.logger.info(f"Invoking model launch CLI command: \n{launch_command}")
        executor_output = self.cli_executor.run_shell_command(launch_args)

        # Extract slurm job id from SLURM CLI output
        launch_output_match = re.match(
            SLURM_JOB_SUBMIT_PATTERN,
            executor_output,
            re.DOTALL,
        )
        if launch_output_match is None:
            raise ValueError(
                "executor_output does not match SLURM_JOB_SUBMIT_PATTERN. \n"
                f"{executor_output}"
            )

        slurm_job_id = launch_output_match.groupdict()["slurm_job_id"]
        return LLMBackend(
            model_name=model_name,
            base_url=None,
            slurm_job_id=slurm_job_id,
        )

    def get_backend_status(self, backend: LLMBackend) -> bool:
        """Verify if backend is still ready or pending, and not preempted."""
        query_command = f"scontrol show job {backend.slurm_job_id}"
        query_args = query_command.split(" ")

        self.logger.info(f"Invoking status check CLI command: \n{query_command}")
        executor_output = self.cli_executor.run_shell_command(query_args)

        # Extract job status from SLURM CLI output
        status_query_match = re.match(
            SLURM_JOB_STATUS_PATTERN,
            executor_output,
            re.DOTALL,
        )

        if status_query_match is None:
            return False

        status = status_query_match.groupdict()["status"]
        return status in VALID_SLURM_STATUSES

    def delete_backend(self, backend: LLMBackend) -> None:
        """Request backend deletion.

        This method should not block and should be invoked only once
        for each job_id.
        """
        deletion_command = f"scancel {backend.slurm_job_id}"
        deletion_args = deletion_command.split(" ")
        self.logger.info(f"Invoking job deletion CLI command: \n{deletion_command}")

        # No need to track command status here.
        self.cli_executor.run_shell_command_detached(deletion_args)
