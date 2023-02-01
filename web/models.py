from enum import Enum
from abc import ABC

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
    state = db.Column(db.Enum(ModelInstanceState), default=ModelInstanceState.LAUNCHING)
    host = db.Column(db.String)
    generations = db.relationship('ModelInstanceGeneration', backref='ModelInstance', lazy=True)

    def get_current_instances():
        return db.select(ModelInstance).filter(
            ModelInstance.state._in(
                [   ModelInstanceState.LAUNCHING,
                    ModelInstanceState.LOADING, 
                    ModelInstanceState.ACTIVE
                ]
            )
        )

    def launch(self):
        model_service = ModelService(self.id, self.name)
        model_service.launch()

    def shutdown(self):
        model_service = ModelService(self.id, self.name, model_host=self.host)
        model_service.shutdown()

    def update_state(self, new_state: ModelInstanceState, new_state_params: dict = {}):
        """Update the state of the model instance and commit to the database

            There is a possibility of a race condition here and we may want 
            to include logic that igore setting previous states.
        """

        if new_state == ModelInstanceState.LOADING:
            self.host = new_state_params["host"]

        self.state = new_state
        
        db.session.add(self)
        db.session.commit()

    def generate(self, prompt: str, username: str, kwargs: dict = {}):

        generation = ModelInstanceGeneration.create(self.id, username, prompt)

        model_service = ModelService(self.id, self.name, model_host=self.host)

        generation_response = model_service.generate(generation.id, prompt, **kwargs)
        generation.response = generation_response

        return generation

    def is_healthy(self):
        model_service = ModelService(self.id, self.name, model_host=self.host)
        model_service.verify_model_instance_health(self.model_state)


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_instance_id = db.Column(db.Integer, db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)
    prompt = db.Column(db.String)
