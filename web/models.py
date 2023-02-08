from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC
import uuid

from flask import current_app
from sqlalchemy.dialects.postgresql import UUID

from errors import InvalidStateError
from db import db, BaseMixin
from services import model_service

MODEL_CONFIG = {
    "OPT-175B": {
        "name": "OPT-175B",
        "description": "175B parameter version of the Open Pre-trained Transformer (OPT) model trained by Meta",
        "url": "https://huggingface.co/meta/opt-175B",
    },
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
        raise InvalidStateError(self)

    def register(self, host: str):
        raise InvalidStateError(self)

    def activate(self):
        raise InvalidStateError(self)

    def shutdown(self):
        raise InvalidStateError(self)

    def is_healthy(self):
        raise InvalidStateError(self)


class PendingState(ModelInstanceState):

    def launch(self):
        try: 
            # ToDo: set job id params here.
            model_service.launch(self._model_instance.id, self._model_instance.name)
            self._model_instance.transition_to_state(ModelInstanceStates.LAUNCHING)
        except:
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        

class LaunchingState(ModelInstanceState):
        
    def register(self, host: str):
        self._model_instance.host = host
        self._model_instance.transition_to_state(ModelInstanceStates.LOADING)


class LoadingState(ModelInstanceState):

    def activate(self):
        self._model_instance.transition_to_state(ModelInstanceStates.ACTIVE)


class ActiveState(ModelInstanceState):

    def generate(self, username, prompt, generation_args):

        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
            prompt=prompt
        )

        # ToDo - add and save response to generation object in db
        return model_service.generate(
            self._model_instance.host, 
            model_instance_generation.id, 
            model_instance_generation.prompt, 
            generation_args
        )

    def is_healthy(self):
        is_healthy = model_service.verify_model_health(self._model_instance.host)
        if not is_healthy:
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy


class FailedState(ModelInstanceState):

    def is_healthy(self):
        return False
        
class CompletedState(ModelInstanceState):

    def is_healthy(self):
        return True
    

class ModelInstanceStates(Enum):
    PENDING = PendingState
    LAUNCHING = LaunchingState
    LOADING = LoadingState
    ACTIVE = ActiveState
    FAILED = FailedState
    COMPLETED = CompletedState


class ModelInstance(BaseMixin, db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid)
    name = db.Column(db.String, nullable=False)
    state = db.Column(db.Enum(ModelInstanceStates), nullable=False, default=(ModelInstanceStates.PENDING))
    host = db.Column(db.String)
    generations = db.relationship('ModelInstanceGeneration', backref='model_instance', lazy=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.state is None:
            self.state = ModelInstanceStates.PENDING
        self._state = self.state.value(self)

    @db.orm.reconstructor
    def init_on_load(self):
        self._state = self.state.value(self)

    @classmethod
    def find_current_instances(cls) -> List[ModelInstance]:
        """Find the current instances of all models"""
        current_instance_query = db.select(cls).filter(cls.state.in_(
                    (   
                        ModelInstanceStates.PENDING,
                        ModelInstanceStates.LAUNCHING,
                        ModelInstanceStates.LOADING, 
                        ModelInstanceStates.ACTIVE
                    )
                )
            )

        return db.session.execute(current_instance_query).scalars().all()

    @classmethod
    def find_current_instance_by_name(cls, name: str) -> Optional[ModelInstance]:
        """Find the current instance of a model by name"""
        current_instance_query = db.select(cls).filter(cls.state.in_(
                    (   
                        ModelInstanceStates.PENDING,
                        ModelInstanceStates.LAUNCHING,
                        ModelInstanceStates.LOADING, 
                        ModelInstanceStates.ACTIVE
                    )
                )
            ).filter_by(name=name) 

        model_instance = db.session.execute(current_instance_query).scalars().first()

        return model_instance

    def transition_to_state(self, new_state: ModelInstanceState):
        """Transition the model instance to a new state"""
        self.state = new_state
        self._state = self.state.value(self)
        self.save()

    def launch(self) -> None:
        self._state.launch()

    def register(self, host: str) -> None:
        self._state.register(host)
    
    def activate(self) -> None:
        self._state.activate()

    def shutdown(self) -> None:
        self._state.shutdown()

    def generate(self, username: str, prompt: str, kwargs: Dict = {}) -> Dict:
        return self._state.generate(username, prompt, kwargs)

    def is_healthy(self) -> bool:
        return self._state.is_healthy()


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_instance_id = db.Column(db.Integer, db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)
    prompt = db.Column(db.String)
