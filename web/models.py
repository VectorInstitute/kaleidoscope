from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC
from datetime import datetime
import uuid

from flask import current_app
from sqlalchemy.dialects.postgresql import UUID

from config import Config
from errors import InvalidStateError
from db import db, BaseMixin
from services import model_service_client


class ModelInstanceState(ABC):
    def __init__(self, model_instance: ModelInstance):
        self._model_instance = model_instance

    def launch(self):
        raise InvalidStateError(self)

    def register(self, host: str):
        raise InvalidStateError(self)

    def verify_activation(self):
        raise InvalidStateError(self)

    def generate(self, username, inputs):
        raise InvalidStateError(self)

    def generate_activations(self, username, prompts, module_names, generation_args):
        raise InvalidStateError(self)

    def get_module_names(self):
        raise InvalidStateError(self)

    def shutdown(self):
        model_service_client.shutdown(self._model_instance.id)
        self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    def is_healthy(self):
        raise InvalidStateError(self)
    
    def is_timed_out(self):
        raise InvalidStateError(self)


class PendingState(ModelInstanceState):
    def launch(self):
        try:
            # ToDo: set job id params here
            model_service_client.launch(
                self._model_instance.id, self._model_instance.name
            )
            self._model_instance.transition_to_state(ModelInstanceStates.LAUNCHING)
        except Exception as err:
            current_app.logger.error(f"Job launch for {self._model_instance.name} failed: {err}")
            self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    def is_healthy(self):
        is_healthy = model_service_client.verify_job_health(self._model_instance.id)
        return is_healthy
    
    def is_timed_out(self):
        return False


class LaunchingState(ModelInstanceState):
    def register(self, host: str):
        self._model_instance.host = host
        self._model_instance.transition_to_state(ModelInstanceStates.LOADING)

    def is_healthy(self):
        return model_service_client.verify_job_health(self._model_instance.id)
    
    def is_timed_out(self):
        False


class LoadingState(ModelInstanceState):
    def verify_activation(self):
        is_active = model_service_client.verify_model_instance_activation(self._model_instance.host, self._model_instance.name)
        if is_active:
            self._model_instance.transition_to_state(ModelInstanceStates.ACTIVE)

    def is_healthy(self):
        return  model_service_client.verify_job_health(self._model_instance.id)
    
    def is_timed_out(self):
        last_event_datetime = self._model_instance.updated_at
        return (datetime.now() - last_event_datetime) > Config.MODEL_INSTANCE_ACTIVATION_TIMEOUT
    

class ActiveState(ModelInstanceState):
    def generate(self, username, inputs):
        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
        )
        model_instance_generation.inputs

        current_app.logger.info(model_instance_generation)

        # ToDo - add and save response to generation object in db
        generation_response = model_service_client.generate(
            self._model_instance.host,
            model_instance_generation.id,
            inputs,
        )
        model_instance_generation.generation = generation_response
        return model_instance_generation

    def get_module_names(self):
        return model_service_client.get_module_names(self._model_instance.host)

    def generate_activations(self, username, inputs):

        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
        )

        current_app.logger.info(model_instance_generation)

        activations_response = model_service_client.generate_activations(
            self._model_instance.host,
            model_instance_generation.id,
            inputs,
        )

        return activations_response

    def is_healthy(self):
        is_healthy = model_service_client.verify_model_health(self._model_instance.name, self._model_instance.host)
        return is_healthy
    
    def is_timed_out(self):
        last_event_datetime = self._model_instance.updated_at
        last_generation = self._model_instance.last_generation()
        if last_generation:
            last_event_datetime = last_generation.created_at

        return (datetime.now() - last_event_datetime) > Config.MODEL_INSTANCE_TIMEOUT


class FailedState(ModelInstanceState):
    def shutdown(self):
        raise InvalidStateError(self)


class CompletedState(ModelInstanceState):
    def shutdown(self):
        raise InvalidStateError(self)


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
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    updated_at = db.Column(
        db.TIMESTAMP, server_default=db.func.now(), onupdate=db.func.current_timestamp()
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
    def find_loading_instances(cls) -> List[ModelInstance]:
        """Find the current instances of all models"""
        current_instance_query = db.select(cls).filter(
            cls.state_name.in_(
                (
                    ModelInstanceStates.LOADING,
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

    def verify_activation(self) -> None:
        self._state.verify_activation()

    def shutdown(self) -> None:
        self._state.shutdown()

    def generate(
        self, username: str, inputs: Dict = {}
    ) -> Dict:
        return self._state.generate(username, inputs)

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

    def is_timed_out(self) -> bool:
        return self._state.is_timed_out()
    
    def last_generation(self):
        last_generation_query = db.select(ModelInstanceGeneration).where(ModelInstanceGeneration.model_instance_id == self.id).order_by(ModelInstanceGeneration.created_at.desc())
        last_generation = db.session.execute(last_generation_query).scalars().first()
        return last_generation
    
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
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    updated_at = db.Column(
        db.TIMESTAMP, server_default=db.func.now(), onupdate=db.func.current_timestamp()
    )

    def serialize(self):
        return {
            "id": str(self.id),
            "model_instance_id": str(self.model_instance_id),
            "prompts": self.prompts,
            "generation": self.generation,
        }