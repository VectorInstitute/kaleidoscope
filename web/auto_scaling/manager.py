"""
Logic for a state shared across web worker threads for auto-scaling.
"""

import logging
import os
import threading
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from random import choice

from .backend_launchers import SLURMCLILauncher, VectorInferenceModelConfig
from .executors import LocalShellCommandExecutor
from .interfaces import LLMBackend, LLMBackendStatus, ModelName, SLURMJobID

MIN_UPDATE_INTERVAL = timedelta(minutes=1.0)
MAX_NUM_HISTORIC_RECORDS = 10
MAX_CONCURRENCY = int(os.environ.get("MAX_SHELL_CONCURRENCY", 10))


class AutoScalingManager:
    """
    Manager for tracking request/usage/performance and auto scaling.
    """

    def __init__(
        self,
        min_update_interval: timedelta = MIN_UPDATE_INTERVAL,
        max_num_historic_records: int = MAX_NUM_HISTORIC_RECORDS,
        max_query_concurrency: int = MAX_CONCURRENCY,
        slurm_qos: str | None = None,
    ):
        self._min_update_interval = min_update_interval

        # Counter for the number of requests to each model since the previous
        # update.
        self._request_counter_lock = threading.Lock()
        self.request_counter: "Counter[ModelName]" = Counter()
        self.previous_update: datetime = datetime.now()

        # Map model name to a deque of average number of requests per second
        # for the previous few seconds.
        self.requests_per_second: dict[ModelName, deque[float]] = defaultdict(
            lambda: deque([0.0], maxlen=max_num_historic_records),
        )

        # Include only backends that are pending or ready.
        # Track SLURM job ID to avoid duplication.
        self._backend_registry_lock = threading.Lock()
        self._backends: dict[SLURMJobID, LLMBackend] = {}
        self._backend_ids_by_model: dict[ModelName, list[SLURMJobID]] = defaultdict(
            list
        )

        cli_executor = LocalShellCommandExecutor()
        self._backend_launcher = SLURMCLILauncher(cli_executor, slurm_qos=slurm_qos)
        self._max_query_concurrency = max_query_concurrency

        self._logger = logging.getLogger(__name__)

    def _update_request_rate_stats(self) -> bool:
        """Update counter if time since previous update exceeds threshold.

        Returns:
            True if updated. Otherwise returns False.
        """
        time_since_update = datetime.now() - self.previous_update
        if time_since_update < self._min_update_interval:
            return False

        self._request_counter_lock.acquire()
        self.previous_update = datetime.now()
        for model_name, request_count in self.request_counter.items():
            requests_per_second = request_count / (time_since_update.microseconds / 1e6)
            self.requests_per_second[model_name].append(requests_per_second)
            self.request_counter[model_name] = 0

        self._request_counter_lock.release()
        return True

    def _refresh_backend_list(self) -> None:
        """Check for preemption in the list of backends."""

    def get_llm_backend(
        self,
        model_name: ModelName,
        log_request: bool = True,
    ) -> LLMBackend | None:
        """Request a backend for a given llm model name.

        Returns None if backend is not available.
        New backends are not launched here- model-launching
        logic should be placed in self.check().


        Logs a request to this model_name unless log_request is False.


        Params:
            model_name: ModelName.
            log_request: bool, whether to log request to the model.

        Returns:
            LLMBackend where is_ready is True unless all
            backends for this model are not ready.

            None if no backend is available at all.
        """
        # Validate model_name
        if VectorInferenceModelConfig.from_model_name(model_name) is None:
            raise ValueError(f"Invalid model name: {model_name}")

        if log_request:
            with self._request_counter_lock:
                self.request_counter[model_name] += 1

        # Return a "ready" backend whenever possible.
        # Otherwise, try to return a pending backend.
        # Return None if there is no pending or ready backend.
        backends = [
            self._backends[backend_id]
            for backend_id in self._backend_ids_by_model[model_name]
        ]
        if len(backends) == 0:
            return None

        backends_ready = [backend for backend in backends if backend.is_ready]
        if len(backends_ready) == 0:
            return backends[0]

        return choice(backends_ready)

    def _deregister_backend(self, slurm_job_id: SLURMJobID) -> None:
        """Delete a backend that is no longer available.

        Invoking this command does not stop the job.

        Example: the backend might be preempted.
        """
        with self._backend_registry_lock:
            for model_name in self._backend_ids_by_model.keys():
                if slurm_job_id in self._backend_ids_by_model[model_name]:
                    self._backend_ids_by_model[model_name].remove(slurm_job_id)

            self._backends.pop(slurm_job_id, None)

    def _update_backend_status(
        self, slurm_job_id: SLURMJobID, status: LLMBackendStatus
    ) -> LLMBackend:
        """Update the status of the given backend.

        Returns the updated backend instance.
        """
        with self._backend_registry_lock:
            backend = self._backends[slurm_job_id]
            backend = backend._replace(status=status)
            self._backends[slurm_job_id] = backend

        return backend

    def add_backend(self, backend: LLMBackend) -> None:
        """Add a LLM backend to registry.

        Params:
            backend: LLMBackend.
                The backend doesn't have to be ready to be added.
                However, job_id must be provided to keep track of the job.
        """
        with self._backend_registry_lock:
            # Avoid adding the same backend twice.
            if backend.slurm_job_id in self._backends:
                return

            self._backend_ids_by_model[backend.model_name].append(backend.slurm_job_id)
            self._backends[backend.slurm_job_id] = backend

    def log_throughput(self, model_name: ModelName, backend_job_id: SLURMJobID):
        """Log a request to the given model *after* a request.

        To be invoked from Flask web worker threads.

        This method is concurrency-safe.

        Params:
            model_name: str.
            backend_job_id: slurm job id.
        """
        raise NotImplementedError

    def check(self) -> None:
        """Update request counter and scale up/down automatically as needed."""
        self._update_request_rate_stats()
        with ThreadPool(self._max_query_concurrency) as thread_pool:
            backends = self._backends.values()
            backend_statuses = list(
                thread_pool.map(self._backend_launcher.get_backend_status, backends),
            )

        self._logger.debug(f"backends: {self._backends}")
        self._logger.debug(f"_backend_ids_by_model: {self._backend_ids_by_model}")
        self._logger.debug(f"requests_per_second: {self.requests_per_second}")

        # De-register all backends that are not valid.
        for backend, backend_status in zip(list(backends), backend_statuses):
            backend = self._update_backend_status(backend.slurm_job_id, backend_status)
            if (not backend.is_ready) and (not backend.is_pending):
                self._logger.info(
                    f"Backend {backend.slurm_job_id} is neither ready nor pending. "
                    f"Status: {backend_status}; Deleting this backend."
                )
                self._backend_launcher.delete_backend(backend)
                self._deregister_backend(backend.slurm_job_id)

        with self._request_counter_lock:
            # Auto-scale-up backends
            for model_name, previous_request_stats in self.requests_per_second.items():
                # TODO: modify to launch additional backends based on threshold.
                if (sum(previous_request_stats) > 0.0) and (
                    len(self._backend_ids_by_model[model_name]) == 0
                ):
                    backend = self._backend_launcher.create_backend(model_name)
                    self.add_backend(backend)

            # Auto-scale-down backends
            models_cleared: list[ModelName] = []
            for model_name, previous_request_stats in self.requests_per_second.items():
                if sum(previous_request_stats) == 0.0:
                    backends = [
                        self._backends.pop(backend_id)
                        for backend_id in self._backend_ids_by_model[model_name]
                    ]
                    for backend in backends:
                        self._logger.info(f"Preempting backend {backend} (inactivity).")
                        self._backend_launcher.delete_backend(backend)

                    self._backend_ids_by_model.pop(model_name)
                    models_cleared.append(model_name)

            for model_name in models_cleared:
                self.requests_per_second.pop(model_name)
