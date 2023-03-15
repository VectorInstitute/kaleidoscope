from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC
import uuid

from flask import current_app
from sqlalchemy.dialects.postgresql import UUID

from config import Config
from errors import InvalidStateError
from db import db, BaseMixin
from services import model_service_client

MODEL_CONFIG = {
    "OPT-175B": {
        "name": "OPT-175B",
        "description": "175B parameter version of the Open Pre-trained Transformer (OPT) model trained by Meta",
        "url": "https://huggingface.co/meta/opt-175B",
        "path": "/models/opt-175b"
    },
    "OPT-6.7B": {
        "name": "OPT-6.7B",
        "description": "6.7B parameter version of the Open Pre-trained Transformer (OPT) model trained by Meta",
        "url": "https://huggingface.co/facebook/opt-6.7b",
        "path": "/models/opt-6.7b"
    },
    "GPT2": {
        "name": "GPT2",
        "description": "GPT2 model trained by OpenAI, available only for testing and development",
        "url": "https://huggingface.co/gpt2",
        "path": "/models/gpt2"
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

    def generate(self, username, prompts, generation_args):
        raise InvalidStateError(self)

    def generate_activations(self, username, prompts, module_names, generation_args):
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
            model_service_client.launch(
                self._model_instance.id, self._model_instance.name, "/scratch/models/gpt2"
            )
            self._model_instance.transition_to_state(ModelInstanceStates.LAUNCHING)
        except Exception as err:
            current_app.logger.error(f"Job launch for {self._model_instance.name} failed: {err}")
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)

    def register(self, host: str):
        # Only the Python job scheduler can register models in pending state
        # since launching happens immediately
        if Config.JOB_SCHEDULER == "python":
            self._model_instance.host = host
            self._model_instance.transition_to_state(ModelInstanceStates.LOADING)
        else:
            raise InvalidStateError(self)

    def is_healthy(self):
        is_healthy = model_service_client.verify_job_health(self._model_instance.id)
        if not is_healthy:
            current_app.logger.error(f"Health check for pending model {self._model_instance.name} failed")
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
            current_app.logger.error(f"Health check for launching model {self._model_instance.name} failed")
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy

    def shutdown(self):
        pass


class LoadingState(ModelInstanceState):
    def activate(self):
        self._model_instance.transition_to_state(ModelInstanceStates.ACTIVE)

    # If we receive multiple registration requests for the same model, just ignore them
    # This will happen whenever a model is loaded onto multiple nodes
    def register(self, host: str):
        pass

    def is_healthy(self):
        is_healthy = model_service_client.verify_job_health(self._model_instance.id)
        if not is_healthy:
            current_app.logger.error(f"Health check for loading model {self._model_instance.name} failed")
            self._model_instance.transition_to_state(ModelInstanceStates.FAILED)
        return is_healthy

    def shutdown(self):
        pass


class ActiveState(ModelInstanceState):

    def generate(self, username, prompts, generation_config):
        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
        )
        model_instance_generation.prompts = prompts
        
        current_app.logger.info(model_instance_generation)

        # ToDo - add and save response to generation object in db
        generation_response = model_service_client.generate(
            self._model_instance.host, 
            model_instance_generation.id, 
            prompts,
            generation_config
        )
        model_instance_generation.generation = generation_response
        return model_instance_generation

    def get_module_names(self):
        return model_service_client.get_module_names(self._model_instance.host)

    def generate_activations(self, username, prompts, module_names, generation_config):

        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
        )

        current_app.logger.info(model_instance_generation)

        activations_response = model_service_client.generate_activations(
            self._model_instance.host, 
            model_instance_generation.id, 
            prompts,
            module_names,
            generation_config,
        )

        return activations_response

    def is_healthy(self):
        is_healthy = model_service_client.verify_model_health(self._model_instance.host)
        if not is_healthy:
            current_app.logger.error(f"Health check for active model {self._model_instance.name} failed")
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
    state_name = db.Column(
        db.Enum(ModelInstanceStates),
        nullable=False,
        default=(ModelInstanceStates.PENDING),
    )
    host = db.Column(db.String)
    generations = db.relationship(
        "ModelInstanceGeneration", backref="model_instance", lazy=True
    )

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
        current_instance_query = db.select(cls).filter(
            cls.state_name.in_(
                (
                    ModelInstanceStates.PENDING,
                    ModelInstanceStates.LAUNCHING,
                    ModelInstanceStates.LOADING,
                    ModelInstanceStates.ACTIVE,
                )
            )
        )

        return db.session.execute(current_instance_query).scalars().all()

    @classmethod
    def find_current_instance_by_name(cls, name: str) -> Optional[ModelInstance]:
        """Find the current instance of a model by name"""
        current_instance_query = (
            db.select(cls)
            .filter(
                cls.state_name.in_(
                    (
                        ModelInstanceStates.PENDING,
                        ModelInstanceStates.LAUNCHING,
                        ModelInstanceStates.LOADING,
                        ModelInstanceStates.ACTIVE,
                    )
                )
            )
            .filter_by(name=name)
        )

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
        current_app.logger.info(f"Received registration request from host {host}")
        self._state.register(host)

    def activate(self) -> None:
        self._state.activate()

    def shutdown(self) -> None:
        self._state.shutdown()

    def generate(
        self, username: str, prompts: List[str], generation_config: Dict = {}
    ) -> Dict:
        return self._state.generate(username, prompts, generation_config)

    def get_module_names(self):
        return self._state.get_module_names()

    def generate_activations(
        self, 
        username: str, 
        prompts: List[str], 
        module_names: List[str], 
        generation_config: Dict = {},
    ) -> Dict:
        return self._state.generate_activations(
            username, prompts, module_names, generation_config
        )

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
    model_instance_id = db.Column(
        UUID(as_uuid=True), db.ForeignKey("model_instance.id")
    )
    username = db.Column(db.String)

    def serialize(self):
        return {
            "id": str(self.id),
            "model_instance_id": str(self.model_instance_id),
            "prompts": self.prompts,
            "generation": self.generation,
        }


# ToDo: Should generalize generation and activation? This needs a design decision.
# class Activation():

#     def serialize(self):
#         return {
#             "model_instance_id": str(self.model_instance_generation_id),
#             "prompt": self.prompt,
#             "generation": self.model_generation
#         }
