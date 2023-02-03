from __future__ import annotations
from enum import Enum, auto
from typing import List, Optional, Dict
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

class ModelInstanceState(ABC):

    def __init__(self, model_instance: ModelInstance):
        self._model_instance = model_instance

    def launch(self):
        raise NotImplementedError(f'Cannot launch model instance in state {self.__class__.__name__}')

    def register(self, host: str):
        raise NotImplementedError(f'Cannot register model instance in state {self.__class__.__name__}')

    def activate(self):
        raise NotImplementedError(f'Cannot activate model instance in state {self.__class__.__name__}')

    def shutdown(self):
        raise NotImplementedError(f'Cannot shutdown model instance in state {self.__class__.__name__}')

    def is_healthy(self):
        raise NotImplementedError(f'Cannot check health of model instance in state {self.__class__.__name__}')


class PendingState(ModelInstanceState):

    def launch(self):
        model_service.dispatch_launch(self._model_instance)
        self._model_instance.transition_to_state(ModelInstanceStates.LAUNCHING)
    

class LaunchingState(ModelInstanceState):

    # def shutdown(self):
    #     model_service.dispatch_shutdown(self._model_instance)
    #     self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    # def is_healthy(self):
    #     pass
    #     #model_service.verify_health(self._model_instance)
        
    def register(self, host: str):
        self._model_instance.host = host
        self._model_instance.transition_to_state(ModelInstanceStates.LOADING)


class LoadingState(ModelInstanceState):

    # def shutdown(self):
    #     model_service.dispatch_job_shutdown(self._model_instance)
    #     self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    def activate(self):
        self._model_instance.transition_to_state(ModelInstanceStates.ACTIVE)

    # def is_healthy(self):
    #     self._model_instance.transition_to_state(ModelInstanceStates.FAILED)

class ActiveState(ModelInstanceState):

    # def shutdown(self):
       
    #     model_service.dispatch_model_shutdown(self._model_instance)
    #     self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    # def is_healthy(self):
    #     pass

    def generate(self, model_instance_generation: ModelInstanceGeneration):
        generation_response = model_service.generate(self._model_instance.host, model_instance_generation)
        model_instance_generation.reponse = generation_response


class FailedState(ModelInstanceState):
    
    def change_state(self, new_state):
        pass


class CompletedState(ModelInstanceState):
    
    def change_state(self, new_state):
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

    def transition_to_state(self, new_state: ModelInstanceState):
        """Update the state of the model instance and commit to the database

            There is a possibility of a race condition here and we may want 
            to include logic that igore setting previous states.
        """
        self.state = new_state
        self._state = ModelInstanceStates[self.state](self)
        self.save()

    def launch(self) -> None:
        self._state.launch()

    def register(self, host: str) -> None:
        self._state.register(host)
    
    def activate(self) -> None:
        self._state.activate()

    def shutdown(self) -> None:
        self._state.shutdown()

    def generate(self, username: str, prompt: str, kwargs: Dict = {}):
        return self._state.generate(username, prompt, kwargs)

    def is_healthy(self):
        return self._state.is_healthy()


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_instance_id = db.Column(db.Integer, db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)
    prompt = db.Column(db.String)
