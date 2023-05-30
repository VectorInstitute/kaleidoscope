import requests

class GatewayServiceClient():

    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port

    def register_model_instance(model_type, master_host, master_port):
        # url = f"http://{self.host}:{self.port}/model_instance"
        # data = {
        #     "model_type": model_type,
        #     "master_host": master_host,
        #     "master_port": master_port
        # }
        # response = requests.post(url, json=data)
        # return response
        pass