import requests
import pytest

# TODO: Configure github runner to server gateway service for testing
# TODO: Complete test cases for all SDK endpoints and find potential opportunities to test with authentication
# TODO: Serve a dummy model class to reinact as a successful model instance and text generation


@pytest.mark.skip(reason="tested on-premise")
def test_ping():
    "Verify the existance of Lingua server"
    server_url = "http://llm.cluster.local:3001/"
    assert requests.get(server_url).ok


@pytest.mark.skip(reason="tested on-premise")
def test_get_models():
    "Verify the existance of Lingua server"
    server_url = "http://llm.cluster.local:3001/models"
    assert requests.get(server_url).ok


@pytest.mark.skip(reason="tested on-premise")
def test_authenticate():
    "Verify the existance of Lingua server"
    server_url = "http://llm.cluster.local:3001/authenticate"
    assert requests.get(server_url).ok


@pytest.mark.skip(reason="tested on-premise")
def test_model_instances():
    "Verify the existance of Lingua server"
    server_url = "http://llm.cluster.local:3001/models/instances"
    assert requests.get(server_url).ok


@pytest.mark.skip(reason="tested on-premise")
def test_create_model_instance():
    "Verify the existance of Lingua server"
    server_url = "http://llm.cluster.local:3001/models/instances"
    assert requests.post(server_url).ok
