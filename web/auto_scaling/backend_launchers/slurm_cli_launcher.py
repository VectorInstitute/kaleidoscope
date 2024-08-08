"""
Logic for launching backend via SLURM CLI.

Note that exact logic for running CLI command is specified
in the shell `executor` implementation.
"""

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

from ..executors import LocalShellCommandExecutor
from ..interfaces import (
    AbstractLLMBackendLauncher,
    AbstractShellCommandExecutor,
    LLMBackend,
    LLMBackendStatus,
    ModelName,
)

# Important: filter out unsafe characters from user-specified "variant".
# Map regexp for model name to Vector Inference "model family", capturing
# Vector Inference "model variant"
MODEL_FAMILIES: dict[str, str] = {
    r"^Mixtral-(?P<variant>[a-zA-Z0-9\.\-]+)$": "mixtral",
    r"^Mistral-(?P<variant>[a-zA-Z0-9\.\-]+)$": "mistral",
    r"^(Meta-)?Llama-3-(?P<variant>[a-zA-Z0-9\.\-]+)$": "llama3",
    r"^(Meta-)?Llama-2-(?P<variant>[a-zA-Z0-9\.\-]+)$": "llama2",
    r"^c4ai-command-r-(?P<variant>[a-zA-Z0-9\.\-]+)$": "command-r",
}
MODEL_VARIANT_PATTERN: str = r"^[a-zA-Z0-9\.\-]+$"
SLURM_JOB_SUBMIT_PATTERN = r".*Submitted batch job (?P<slurm_job_id>\d+)([\s\n].*)?"
SLURM_JOB_STATUS_PATTERN = r".*JobState=(?P<status>\w+)\s.+"
VALID_BACKEND_STATUSES = ["READY"]


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
        vector_inference_venv_base_path: str | None = None,
        slurm_qos: str | None = None,
    ):
        """
        Optionally, provide cli_executor instance.

        Otherwise, a default one would be selected.

        Params:
            vector_inference_venv_base_path: path to Vector Inference venv
                e.g., if "activate" is at "/home/llm/env/bin/activate",
                set vector_inference_venv_base_path to "/home/llm/env/".
            slurm_qos: optionally, specify SLURM qos for launching jobs.
                Forwarded to Vector-Inference.
        """
        self.logger = logging.getLogger(__name__)

        if cli_executor is None:
            cli_executor = LocalShellCommandExecutor()
        self.cli_executor = cli_executor

        if vector_inference_venv_base_path is None:
            current_dir = Path(__file__).resolve()
            self.logger.info(f"current_dir: {current_dir}")
            self._venv_base_path = Path.joinpath(
                current_dir.parents[3],
                "env",
            )
        else:
            self._venv_base_path = Path(vector_inference_venv_base_path)

        python_path = Path.joinpath(self._venv_base_path, "bin", "python")
        vec_inf_script_path = Path.joinpath(self._venv_base_path, "bin", "vec-inf")
        self.base_vec_inf_args = (
            python_path.as_posix(),
            vec_inf_script_path.as_posix(),
        )
        self.slurm_qos = slurm_qos

    def _get_args_from_command(self, vec_inf_command: str) -> list[str]:
        """Given vec_inf command (e.g., status job_id), return full CLI arg list."""
        return [*self.base_vec_inf_args, *vec_inf_command.split(" ")]

    def create_backend(self, model_name: ModelName) -> LLMBackend:
        """Request Backend creation.

        Returns:
            LLMBackend, where slurm_job_id is provided but not base_url.
        """
        model_config = VectorInferenceModelConfig.from_model_name(model_name)
        if model_config is None:
            raise ValueError(f"{model_name} is not known.")

        # must validate and filter out shell escapes.
        assert re.match(MODEL_VARIANT_PATTERN, model_config.model_variant) is not None
        launch_command = (
            f"launch {model_config.model_family}"
            f" --model-variant {model_config.model_variant}"
        )
        if self.slurm_qos is not None:
            launch_command += f" --qos {self.slurm_qos}"

        launch_args = self._get_args_from_command(launch_command)
        _full_launch_command = " ".join(launch_args)
        self.logger.info(
            f"Invoking model backend launch command: \n{_full_launch_command}"
        )

        executor_output = self.cli_executor.run_shell_command(launch_args)

        # Extract slurm job id from Vector Inference output
        launch_output_data = json.loads(executor_output)
        slurm_job_id = str(int(launch_output_data["slurm_job_id"]))
        return LLMBackend(
            model_name=model_name,
            is_pending=True,
            status=LLMBackendStatus(base_url=None),
            slurm_job_id=slurm_job_id,
        )

    def get_backend_status(self, backend: LLMBackend) -> LLMBackendStatus:
        """Return LLM Backend status and API URL if applicable.

        Returns:
            LLMBackendStatus.
        """
        query_command = f"status --json-mode {backend.slurm_job_id}"
        query_args = self._get_args_from_command(query_command)
        _full_query_command = " ".join(query_args)
        self.logger.debug(
            f"Invoking model backend status check command: \n{_full_query_command}"
        )

        executor_output = self.cli_executor.run_shell_command(query_args)
        backend_status_data = json.loads(executor_output)
        backend_base_url: str | None = backend_status_data["base_url"]

        # Apply heuristics to set backend_base_url to None if invalid.
        if (backend_base_url is not None) and (not backend_base_url.startswith("http")):
            backend_base_url = None

        return LLMBackendStatus(
            base_url=backend_base_url,
            raw_status_data=backend_status_data,
        )

    def delete_backend(self, backend: LLMBackend) -> None:
        """Request backend deletion.

        This method should not block and should be invoked only once
        for each job_id.
        """
        deletion_command = f"shutdown {backend.slurm_job_id}"
        deletion_args = self._get_args_from_command(deletion_command)
        _full_deletion_command = " ".join(deletion_args)
        self.logger.info(
            f"Invoking model backend deletion command: \n{_full_deletion_command}"
        )

        # No need to track command status here.
        self.cli_executor.run_shell_command_detached(deletion_args)
