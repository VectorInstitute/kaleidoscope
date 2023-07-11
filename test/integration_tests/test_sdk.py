# Leverage file to test concurrently with Kscope SDK version releases
import kscope
import pytest
import socket
import time

hostname = socket.gethostname()

# NOTE: WIP
@pytest.mark.skipif(hostname != "llm", reason="tests for on-premise only")
class TestClient:
    # def test_authenticate(self):
    #     client = kscope.Client("localhost", 3001)
    #     auth_key = client.authenticate()
    #     assert isinstance(auth_key, str)

    def test_get_models(self):
        client = kscope.Client("localhost", 3001)
        models = client.models
        assert isinstance(models, list)

    def test_create_model_instance(self):
        client = kscope.Client("localhost", 3001)
        model_instance = client.load_model("OPT-6.7B")
        while model_instance.state != "ACTIVE":
            time.sleep(1)
        assert isinstance(model_instance, kscope.Model)

    def test_generate(self):
        client = kscope.Client("localhost", 3001)
        model_instance = client.load_model("OPT-6.7B")
        while model_instance.state != "ACTIVE":
            time.sleep(1)
        prompts = ["test prompt 1", "test prompt 2"]
        generation_config = {"temperature": 0.5}
        response = model_instance.generate(prompts, generation_config)
        assert isinstance(response.generation, dict)