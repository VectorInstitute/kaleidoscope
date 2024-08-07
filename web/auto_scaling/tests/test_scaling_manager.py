"""
Single-thread tests for the auto-scaling manager.
"""

import time
from datetime import datetime, timedelta

import pytest

from web.auto_scaling.manager import AutoScalingManager, LLMBackend

MODEL_AVAILABLE_URL = "http://model_available:8000/v1"
MODEL_AVAILABLE_URL_ALT = "http://model_available_alt:8000/v1"
NUM_REPETITIONS = 1000
UPDATE_INTERVAL = 0.1


@pytest.fixture()
def scaling_manager_empty() -> AutoScalingManager:
    return AutoScalingManager(min_update_interval=timedelta(seconds=UPDATE_INTERVAL))


@pytest.fixture()
def scaling_manager(scaling_manager_empty: AutoScalingManager) -> AutoScalingManager:
    scaling_manager_empty._backends = {
        "1000": LLMBackend("Mistral-available", MODEL_AVAILABLE_URL, "1000"),
        "1001": LLMBackend("Mistral-available", None, "1001"),
        "1010": LLMBackend("Mistral-pending", None, "1010"),
    }
    for index, backend in scaling_manager_empty._backends.items():
        scaling_manager_empty._backend_ids_by_model[backend.model_name].append(index)

    return scaling_manager_empty


@pytest.mark.parametrize(("model_name",), [("Mistral-pending",), ("Mistral-other",)])
def test_get_backend_none_available(scaling_manager, model_name: str):
    """Test retrieving backend for a model where no backend is ready."""
    output = scaling_manager.get_llm_backend(model_name)
    assert output is None


def test_get_backend_mixed_availability(scaling_manager):
    """Test retrieving backend for a model where only 1 of 2 backends is ready."""
    # this test is non-deterministic.
    for _ in range(NUM_REPETITIONS):
        output = scaling_manager.get_llm_backend("Mistral-available")
        assert output is not None
        assert output.base_url == MODEL_AVAILABLE_URL

    assert scaling_manager.request_counter["Mistral-available"] == NUM_REPETITIONS


def test_update_request_metrics(scaling_manager):
    """Test updating model request throughput metrics."""
    start_time = datetime.now()
    for _ in range(NUM_REPETITIONS):
        scaling_manager.get_llm_backend("Mistral-available")

    time.sleep(UPDATE_INTERVAL)
    actual_time_elapsed = datetime.now() - start_time
    request_rate_reference = NUM_REPETITIONS / (actual_time_elapsed.microseconds / 1e6)
    scaling_manager._update_request_rate_stats()
    request_rate = scaling_manager.requests_per_second["Mistral-available"][-1]

    assert abs(request_rate_reference - request_rate) < (request_rate_reference / 10)


def test_register_backend_url(scaling_manager):
    """Test setting backend URL after requesting one."""
    scaling_manager.set_backend_url("1010", MODEL_AVAILABLE_URL)
    llm_backend_updated = scaling_manager.get_llm_backend("Mistral-pending")

    assert llm_backend_updated.base_url == MODEL_AVAILABLE_URL
