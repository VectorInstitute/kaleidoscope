"""Module for model configurations"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC
from datetime import datetime
from db import db, BaseMixin
from flask import current_app
from sqlalchemy.dialects.postgresql import UUID
import uuid

from config import Config
from errors import InvalidStateError
from services import model_service_client


AVAIALBLE_MODELS = model_service_client.get_available_models()


class ModelInstanceState(ABC):
    """Class for a model instance state"""

    def __init__(self, model_instance: ModelInstance):
        self._model_instance = model_instance

    def launch(self):
        """Launch a new instance of a model"""
        raise InvalidStateError(self)

    def register(self, host: str):
        """Register a model instance"""
        raise InvalidStateError(self)

    def verify_activation(self):
        """Check if a model is active and ready to service requests"""
        raise InvalidStateError(self)

    def generate(self, username, inputs):
        """Send a generation request to a model"""
        raise InvalidStateError(self)

    def shutdown(self):
        """Shutdown a model"""
        model_service_client.shutdown(self._model_instance.id)
        self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    def is_healthy(self):
        """Check if a model is healthy"""
        raise InvalidStateError(self)

    def is_timed_out(self):
        raise InvalidStateError(self)
    

class PendingState(ModelInstanceState):
    """Class for model pending state"""

    def launch(self):
        current_app.logger.info(f"Issuing launch command for model {self._model_instance.name}")
        """Launch a model"""
        try:
            # ToDo: set job id params here
            model_service_client.launch(
                self._model_instance.id,
                self._model_instance.name,
            )
            self._model_instance.transition_to_state(ModelInstanceStates.LAUNCHING)
        except Exception as err:
            current_app.logger.error(f"Job launch for {self._model_instance.name} failed: {err}")
            self._model_instance.transition_to_state(ModelInstanceStates.COMPLETED)

    def is_healthy(self):
        """Determine model health status"""
        return model_service_client.verify_job_health(self._model_instance.id)

    def is_timed_out(self):
        return False


class LaunchingState(ModelInstanceState):
    """Class for model launching state"""

    def register(self, host: str):
        """Register model as loading"""
        self._model_instance.host = host
        self._model_instance.transition_to_state(ModelInstanceStates.LOADING)

    def is_healthy(self):
        """Retrieve model health status"""
        return model_service_client.verify_job_health(self._model_instance.id)

    def is_timed_out(self):
        return False


class LoadingState(ModelInstanceState):
    """Class for model loading state"""

    def verify_activation(self):
        is_active = model_service_client.verify_model_instance_active(self._model_instance.host, self._model_instance.name)
        if is_active:
            self._model_instance.transition_to_state(ModelInstanceStates.ACTIVE)

    def is_healthy(self):
        return model_service_client.verify_job_health(self._model_instance.id)
    
    def is_timed_out(self):
        last_event_datetime = self._model_instance.updated_at
        return (datetime.now() - last_event_datetime) > Config.MODEL_INSTANCE_ACTIVATION_TIMEOUT


class ActiveState(ModelInstanceState):
    """Class for model active state"""

    def generate(self, username, inputs):
        model_instance_generation = ModelInstanceGeneration.create(
            model_instance_id=self._model_instance.id,
            username=username,
        )
        model_instance_generation.prompts = inputs["prompts"]

        # ToDo - add and save response to generation object in db
        generation_response = model_service_client.generate(
            self._model_instance.host,
            self._model_instance.name,
            inputs
        )
        model_instance_generation.generation = generation_response
        return model_instance_generation

    def is_healthy(self):
        return model_service_client.verify_model_health(self._model_instance.host, self._model_instance.name)
    
    def is_timed_out(self):
        last_event_datetime = self._model_instance.updated_at
        last_generation = self._model_instance.last_generation()
        if last_generation:
            last_event_datetime = last_generation.created_at

        return (datetime.now() - last_event_datetime) > Config.MODEL_INSTANCE_TIMEOUT


class FailedState(ModelInstanceState):
    """Class for failed model instance state"""

    def shutdown(self):
        raise InvalidStateError(self)


class CompletedState(ModelInstanceState):
    """Class for completed model instance state"""

    def shutdown(self):
        raise InvalidStateError(self)


class ModelInstanceStates(Enum):
    """Class for model instance states"""

    PENDING = PendingState
    LAUNCHING = LaunchingState
    LOADING = LoadingState
    ACTIVE = ActiveState
    FAILED = FailedState
    COMPLETED = CompletedState


class ModelInstance(BaseMixin, db.Model):
    """Class for model instances"""

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String, nullable=False)
    state_name = db.Column(
        db.Enum(ModelInstanceStates),
        nullable=False,
        default=(ModelInstanceStates.PENDING),
    )
    host = db.Column(db.String)
    generations = db.relationship("ModelInstanceGeneration", backref="model_instance", lazy=True)
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    updated_at = db.Column(
        db.TIMESTAMP,
        server_default=db.func.now(),
        onupdate=db.func.current_timestamp(),
    )

    def __init__(self, **kwargs):
        """Initialize model instance state"""
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
        """Launch model"""
        self._state.launch()

    def register(self, host: str) -> None:
        """Register model"""
        current_app.logger.info(f"Received registration request from host {host}")
        self._state.register(host)

    def verify_activation(self) -> None:
        self._state.verify_activation()

    def shutdown(self) -> None:
        """Shutdown model"""
        self._state.shutdown()

    def generate(self, username: str, inputs: Dict) -> Dict:
        return self._state.generate(username, inputs)

    def is_healthy(self) -> bool:
        """Retrieve health status"""
        return self._state.is_healthy()

    def is_timed_out(self):
        return self._state.is_timed_out()

    def last_generation(self):
        last_generation_query = (
            db.select(ModelInstanceGeneration)
            .where(ModelInstanceGeneration.model_instance_id == self.id)
            .order_by(ModelInstanceGeneration.created_at.desc())
        )
        last_generation = db.session.execute(last_generation_query).scalars().first()
        return last_generation

    def serialize(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "state": self.state_name.name,
        }


class ModelInstanceGeneration(BaseMixin, db.Model):
    """Class for generating model instances from DB"""

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_instance_id = db.Column(UUID(as_uuid=True), db.ForeignKey("model_instance.id"))
    username = db.Column(db.String)
    created_at = db.Column(db.TIMESTAMP, server_default=db.func.now())
    updated_at = db.Column(
        db.TIMESTAMP,
        server_default=db.func.now(),
        onupdate=db.func.current_timestamp(),
    )

    def serialize(self):
        return {
            "id": str(self.id),
            "model_instance_id": str(self.model_instance_id),
            "prompts": self.prompts,
            "generation": self.generation,
        }
