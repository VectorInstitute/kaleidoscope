from __future__ import annotations
from enum import Enum, auto
from typing import List, Optional
from abc import ABC, abstractmethod

from db import db, BaseMixin
from services import model_service

MODEL_CONFIG = {
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

class Model():

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    def launch(self, model_instance: ModelInstance) -> bool:
        
        success = model_service.launch(model_instance)
        return success


class ModelInstanceState(ABC):

    def launch():
        pass
    
    def shutdown():
        pass

    def is_healthy():
        pass

    def change_state(new_state):
        pass


class PendingState(ModelInstanceState):

    def launch():
        pass

    def is_healthy():
        pass

    def change_state(new_state):
        pass


class LaunchingState(ModelInstanceState):

    def launch():
        pass

    def shutdown():
        pass

    def is_healthy():
        pass

    def change_state(new_state):
        pass


class LoadingState(ModelInstanceState):

    def shutdown():
        pass

    def is_healthy():
        pass

    def change_state(new_state):
        pass


class ActiveState(ModelInstanceState):

    def shutdown():
        pass

    def is_healthy():
        pass

    def change_state(new_state):
        pass


class FailedState(ModelInstanceState):

    def is_healthy():
        pass

    def change_state(new_state):
        pass


class CompletedState(ModelInstanceState):

    def is_healthy():
        pass

    def change_state(new_state):
        pass

class ModelInstanceStates(Enum):
    PENDING = PendingState
    LAUNCHING = LaunchingState
    LOADING = LoadingState
    ACTIVE = ActiveState
    FAILED = FailedState
    COMPLETED = CompletedState

class ModelInstance(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    state = db.Column(db.Enum(ModelInstanceStates), default=ModelInstanceStates.PENDING)
    host = db.Column(db.String)
    generations = db.relationship('ModelInstanceGeneration', backref='model_instance', lazy=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = ModelInstanceStates[self.state]()

    @classmethod
    def find_current_instances(cls) -> List[ModelInstance]:
        current_instance_query = db.select(cls.state.in_(
                    [   ModelInstanceStates.LAUNCHING,
                        ModelInstanceStates.LOADING, 
                        ModelInstanceStates.ACTIVE
                    ]
                ))

        current_instances = db.session.execute(current_instance_query).all()

        return current_instances

    @classmethod
    def find_current_instance_by_name(cls, name: str) -> Optional[ModelInstance]:
        current_instance_query = db.select(
            cls.state.in_(
                [   ModelInstanceStates.LAUNCHING,
                    ModelInstanceStates.LOADING, 
                    ModelInstanceStates.ACTIVE
                ]
            )
        ).filter_by(name=name)

        model_instance = db.session.execute(current_instance_query).first()

        return model_instance

    def launch(self) -> None:
        self._state.launch()
        #model_service.launch(self)

    def shutdown(self) -> None:
        self._state.shutdown()
        #model_service.shutdown(self)

    def update_state(self, new_state: ModelInstanceState):
        """Update the state of the model instance and commit to the database

            There is a possibility of a race condition here and we may want 
            to include logic that igore setting previous states.
        """
        self.state = new_state
        self._state = ModelInstanceStates[self.state]()
        self._state.context = self
        self.save()

    def generate(self, prompt: str, username: str, kwargs: dict = {}):
        return self._state.generate(prompt, username, kwargs)

        # generation = ModelInstanceGeneration.create(self.id, username, prompt)

        # generation_response = model_service.generate(self, generation.id, prompt, **kwargs)
        # generation.response = generation_response

    def is_healthy(self):
        return self._state.is_healthy()
        # model_service = ModelService(self.id, self.name, model_host=self.host)
        # model_service.verify_model_instance_health(self.model_state)


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_instance_id = db.Column(db.Integer, db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)
    prompt = db.Column(db.String)
