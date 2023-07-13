"""Unit tests for a mock gateway service"""
import requests
import pytest
from urllib.parse import urljoin
import os
import socket

# TODO: Configure github runner to server gateway service for testing
# TODO: Complete test cases for all SDK endpoints and find potential
#       opportunities to test with authentication
# TODO: Serve a dummy model class to reinact as a successful model instance and text generation

class TestClient:

@pytest.mark.skip(reason="tested on-premise")
def test_ping():
    "Verify the existance of Kaleidoscope server"
    server_url = "http://llm.cluster.local:3001/"
    assert requests.get(server_url, timeout=300).ok

    def test_ping(self):
        "Verify the existance of Kaleidoscope server"
        assert requests.get(self.server_url).ok

@pytest.mark.skip(reason="tested on-premise")
def test_get_models():
    "Verify the existance of Kaleidoscope server"
    server_url = "http://llm.cluster.local:3001/models"
    assert requests.get(server_url, timeout=300).ok

    def test_authenticate(self):
        "Verify the existance of Kaleidoscope server"
        url = urljoin(self.server_url, "authenticate")
        assert requests.get(url).status_code == 405

@pytest.mark.skip(reason="tested on-premise")
def test_authenticate():
    "Verify the existance of Kaleidoscope server"
    server_url = "http://llm.cluster.local:3001/authenticate"
    assert requests.get(server_url, timeout=300).ok

    def test_route_playground(self):
        "Verify the existance of playground route on Kaleidoscope client"
        url = urljoin(self.server_url, "playground")
        assert requests.get(url).ok

@pytest.mark.skip(reason="tested on-premise")
def test_model_instances():
    "Verify the existance of Kaleidoscope server"
    server_url = "http://llm.cluster.local:3001/models/instances"
    assert requests.get(server_url, timeout=300).ok


@pytest.mark.skip(reason="tested on-premise")
def test_create_model_instance():
    "Verify the existance of Kaleidoscope server"
    server_url = "http://llm.cluster.local:3001/models/instances"
    assert requests.post(server_url, timeout=300).ok
