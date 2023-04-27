import requests
import pytest
from urllib.parse import urljoin
import os
import socket


class TestClient:

    server_url = "http://localhost:3001/"

    def test_ping(self):
        "Verify the existance of Kaleidoscope server"
        assert requests.get(self.server_url).ok

    def test_get_models(self):
        "Verify Kaleidoscope models"
        url = urljoin(self.server_url, "models")
        assert requests.get(url).ok

    def test_authenticate(self):
        "Verify the existance of Kaleidoscope server"
        url = urljoin(self.server_url, "authenticate")
        assert requests.get(url).status_code == 405

    def test_model_instances(self):
        "Verify Kaleidoscope model instances"
        url = urljoin(self.server_url, "models/instances")
        assert requests.get(url).ok

    def test_route_playground(self):
        "Verify the existance of playground route on Kaleidoscope client"
        url = urljoin(self.server_url, "playground")
        assert requests.get(url).ok

    def test_route_reference(self):
        "Verify the existance of reference route on Kaleidoscope client"
        url = urljoin(self.server_url, "reference")
        assert requests.get(url).ok