import json
from unittest.mock import patch
import pytest
import os
import sys
import socket

# FIXME: config import errors
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../web"))
)
# from gateway_service import create_app
# from models import ModelInstance

hostname = socket.gethostname()

# NOTE: WIP
@pytest.mark.skipif(hostname != "llm", reason="tests for on-premise only")
class TestModelInstances:
    # @pytest.fixture
    # def client():
    #     app = create_app()
    #     app.config["TESTING"] = True
    #     with app.test_client() as client:
    #         with app.app_context():
    #             ModelInstance.create(name="test_model_instance")
    #             yield client

    # @pytest.fixture
    # def access_token():
    #     response = client.post(
    #         "/auth/login",
    #         data=json.dumps({"username": "test_user", "password": "test_password"}),
    #         headers={"Content-Type": "application/json"},
    #     )

    #     return response.json["access_token"]

    def test_get_models(client):
        response = client.get("/models/")
        assert response.status_code == 200
        assert response.json == ["model1", "model2"]

    def test_get_current_model_instances(client):
        response = client.get("/models/instances")
        assert response.status_code == 200

    def test_create_model_instance(client, access_token):
        response = client.post(
            "/models/instances",
            data=json.dumps({"name": "test_model_instance"}),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        assert response.status_code == 201
        assert response.json["name"] == "test_model_instance"

    @patch("models.ModelInstance.launch")
    def test_create_model_instance_launch(mock_launch, client, access_token):
        response = client.post(
            "/models/instances",
            data=json.dumps({"name": "test_model_instance"}),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        mock_launch.assert_called()

    def test_get_model_instance(client, access_token):
        model_instance = ModelInstance.find_by_name("test_model_instance")
        response = client.get(
            f"/models/instances/{model_instance.id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        assert response.json["name"] == "test_model_instance"

    def test_remove_model_instance(client, access_token):
        model_instance = ModelInstance.find_by_name("test_model_instance")
        response = client.delete(
            f"/models/instances/{model_instance.id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        assert response.json["name"] == "test_model_instance"

    @patch("models.ModelInstance.register")
    def test_register_model_instance(mock_register, client, access_token):
        model_instance = ModelInstance.find_by_name("test_model_instance")
        response = client.post(
            f"/models/instances/{model_instance.id}/register",
            data=json.dumps({"host": "test_host"}),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        mock_register.assert_called()

    def test_activate_model_instance(client, access_token):
        model_instance = ModelInstance.find_by_name("test_model_instance")
        response = client.post(
            f"/models/instances/{model_instance.id}/activate",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        assert response.json["name"] == "test_model_instance"