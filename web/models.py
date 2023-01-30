from enum import Enum

from db import db, BaseMixin
from services import ModelService

MODELS = {
    "OPT-175B": {
        "name": "OPT-175B",
        "description": "175B parameter version of the Open Pre-trained Transformer (OPT) model trained by Meta",
        "url": "https://huggingface.co/meta/opt-175B",
    },
    # "OPT-66B": {
    #     "name": "OPT-66B",
    #     "description": "66B parameter version of the Open Pre-trained Transformer (OPT) model trained by Meta",
    #     "url": "https://huggingface.co/meta/opt-66B",  
    # },
    # "Galactica-120B": {
    #     "name": "Galactica-120B",
    #     "description": "120B parameter version of the Galactica model trained by Meta",
    #     "url": "https://huggingface.co/meta/galactica-120B",
    # }
}

class ModelInstanceState(Enum):
    LAUNCHING = 0
    LOADING = 1
    ACTIVE = 2
    FAILED = 3
    COMPLETED = 4


class ModelInstance(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    host = db.Column(db.String)
    state = db.Column(db.Enum(ModelInstanceState), default=ModelInstanceState.LAUNCHING)
    generations = db.relationship('ModelInstanceGeneration', backref='ModelInstance', lazy=True)

    def __init__(self, name, host):
        self.name = name
        self.host = host
        self.model_service = ModelService(self.id, self.name)

    def get_current_instances():
        return db.select(ModelInstance).filter(ModelInstance.state._in([ModelInstanceState.LAUNCHING, ModelInstanceState.LOADING, ModelInstanceState.ACTIVE]))

    def launch(self):
        self.model_service.launch()

    def shutdown(self):
        self.model_service.shutdown()

    def generate(self, prompt, username, **kwargs):

        generation = ModelInstanceGeneration.create(self.id, username, prompt)

        generation_response = self.model_service.generate(generation.id, prompt, **kwargs)
        generation.response = generation_response

        return generation

    def is_healthy(self):
        try:
            health_response = self.model_service.verify_model_health()
            response = requests.get(self.base_addr + "/health")
            return response.status_code == 200
        except:
            return False


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_instance_id = db.Column(db.Integer, db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)
    prompt = db.Column(db.String)

    def __init__(
        self,
        model_instance_id,
        username,
        prompt
    ):
        self.model_instance_id = model_instance_id
        self.username = username
        self.prompt = prompt
