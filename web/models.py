from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC
import uuid

from flask import current_app
from sqlalchemy.dialects.postgresql import UUID

from errors import InvalidStateError
from db import db, BaseMixin
from services import model_service_client

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
    # }q
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

    def generate(self, username, prompt, generation_args):
        raise InvalidStateError(self)

    def generate_activations(self, username, prompt, generation_args):
        raise InvalidStateError(self)

    def get_module_names(self):
        raise InvalidStateError(self)

    def shutdown(self):
        raise InvalidStateError(self)

    def is_healthy(self):
        raise InvalidStateError(self)


class PendingState(ModelInstanceState):

    def launch(self):
        try: 
            # ToDo: set job id params here
            model_service_client.launch(self._model_instance.id, self._model_instance.name, "/model_path")
            self._model_instance.transition_to_state(ModelInstanceStates.LAUNCHING)
        except Exception as err:
            current_app.logger.error(f"Job launch failed: {err}")
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)

    def is_healthy(self):
        is_healthy = model_service_client.verify_job_health(self._model_instance.id)
        if not is_healthy:
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy

    def shutdown(self):
        pass
        

class LaunchingState(ModelInstanceState):
        
    def register(self, host: str):
        self._model_instance.host = host
        self._model_instance.transition_to_state(ModelInstanceStates.LOADING)

    def is_healthy(self):
        is_healthy = model_service_client.verify_job_health(self._model_instance.id)
        if not is_healthy:
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy

    def shutdown(self):
        pass


class LoadingState(ModelInstanceState):

    def activate(self):
        self._model_instance.transition_to_state(ModelInstanceStates.ACTIVE)

    def is_healthy(self):
        is_healthy = model_service_client.verify_job_health(self._model_instance.id)
        if not is_healthy:
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy

    def shutdown(self):
        pass


class ActiveState(ModelInstanceState):

    def generate(self, username, prompt, generation_config):

        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
        )
        model_instance_generation.prompt = prompt
        
        current_app.logger.info(model_instance_generation)

        # ToDo - add and save response to generation object in db
        generation_response = model_service_client.generate(
            self._model_instance.host, 
            model_instance_generation.id, 
            prompt,
            generation_config
        )
        model_instance_generation.generation = generation_response
        return model_instance_generation

    def get_module_names(self):
        return model_service_client.get_module_names(self._model_instance.host)

    def generate_activations(self, username, prompt, generation_args):

        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
            prompt=prompt
        )

        # ToDo - add and save response to generation object in db
        return model_service_client.generate_activations(
            self._model_instance.host, 
            model_instance_generation.id, 
            model_instance_generation.prompt, 
            generation_args
        )

    def is_healthy(self):
        is_healthy = model_service_client.verify_model_health(self._model_instance.host)
        if not is_healthy:
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy

    def shutdown(self):
        pass


class FailedState(ModelInstanceState):

    def is_healthy(self):
        return False

    def shutdown(self):
        pass

        
class CompletedState(ModelInstanceState):

    def is_healthy(self):
        return True

    def shutdown(self):
        pass
    

class ModelInstanceStates(Enum):
    PENDING = PendingState
    LAUNCHING = LaunchingState
    LOADING = LoadingState
    ACTIVE = ActiveState
    FAILED = FailedState
    COMPLETED = CompletedState


class ModelInstance(BaseMixin, db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String, nullable=False)
    state_name = db.Column(db.Enum(ModelInstanceStates), nullable=False, default=(ModelInstanceStates.PENDING))
    host = db.Column(db.String)
    generations = db.relationship('ModelInstanceGeneration', backref='model_instance', lazy=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.state_name is None:
            self.state_name = ModelInstanceStates.PENDING
        self._state = self.state_name.value(self)

    @db.orm.reconstructor
    def init_on_load(self):
        self._state = self.state_name.value(self)

    @classmethod
    def find_current_instances(cls) -> List[ModelInstance]:
        """Find the current instances of all models"""
        current_instance_query = db.select(cls).filter(cls.state_name.in_(
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
        current_instance_query = db.select(cls).filter(cls.state_name.in_(
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

    def transition_to_state(self, new_state: ModelInstanceStates):
        """Transition the model instance to a new state"""
        self.state_name = new_state
        self._state = self.state_name.value(self)
        self.save()

    def launch(self) -> None:
        self._state.launch()

    def register(self, host: str) -> None:
        current_app.logger.info(f"[register] called, host={host}")
        self._state.register(host)
    
    def activate(self) -> None:
        self._state.activate()

    def shutdown(self) -> None:
        self._state.shutdown()

    def generate(self, username: str, prompt: str, generation_config: Dict = {}) -> Dict:
        return self._state.generate(username, prompt, generation_config)

    def get_module_names(self):
        return self._state.get_module_names()

    def generate_activations(self, username: str, prompt: str, kwargs: Dict = {}) -> Dict:
        return self._state.generate_activations(username, prompt, kwargs)

    def is_healthy(self) -> bool:
        return self._state.is_healthy()

    def serialize(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "state": self.state_name.name,
        }


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_instance_id = db.Column(UUID(as_uuid=True), db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)

    def serialize(self):
        return {
            "id": str(self.id),
            "model_instance_id": str(self.model_instance_id),
            "prompt": self.prompt,
            "generation": self.generation
        }

# ToDo: Should generalize generation and activation? This needs a design decision.
# class Activation():

#     def serialize(self):
#         return {
#             "model_instance_id": str(self.model_instance_generation_id),
#             "prompt": self.prompt,
#             "generation": self.model_generation
#         }