import pytest
from unittest.mock import Mock, patch
import os
import sys
import socket

# FIXME: config import errors
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../web"))
)
# from services.model_service_client import launch, generate, verify_model_health, verify_job_health

hostname = socket.gethostname()

# NOTE: WIP
@pytest.mark.skipif(hostname != "llm", reason="tests for on-premise only")
class TestModelClient:

    # @pytest.fixture
    # def mock_current_app():
    #     with patch('main.current_app') as mock_app:
    #         yield mock_app

    def test_launch(mock_current_app):
        mock_current_app.logger = Mock()
        launch("instance_id", "model_name", "model_path")
        mock_current_app.logger.info.assert_called_with(
            "Preparing to launch model: model_name"
        )

    def test_generate(mock_current_app):
        mock_current_app.logger = Mock()
        response_json = {"response_key": "response_value"}
        with patch("main.requests.post") as mock_post:
            mock_post.return_value.json.return_value = response_json
            result = generate(
                "host", 1, ["prompt1", "prompt2"], {"config_key": "config_value"}
            )
            mock_current_app.logger.info.assert_called_with(
                "Sending generation request to http://host/generate"
            )
            mock_post.assert_called_with(
                "http://host/generate",
                json={"prompt": ["prompt1", "prompt2"], "config_key": "config_value"},
            )
            assert result == response_json

    def test_get_module_names(mock_current_app):
        mock_current_app.logger = Mock()
        response_json = {"response_key": "response_value"}
        with patch("main.requests.get") as mock_get:
            mock_get.return_value.json.return_value = response_json
            result = get_module_names("host")
            mock_get.assert_called_with("http://host/module_names")
            assert result == response_json

    def test_verify_model_health(mock_current_app):
        with patch("main.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            assert verify_model_health("host") == True
            mock_get.return_value.status_code = 500
            assert verify_model_health("host") == False

    def test_verify_job_health():
        with patch("main.subprocess.check_output") as mock_check_output:
            mock_check_output.return_value = b"output from ssh command"
            assert verify_job_health("instance_id") == True
            mock_check_output.return_value = b""
            assert verify_job_health("instance_id") == False
