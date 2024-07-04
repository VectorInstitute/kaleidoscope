"""A module to abstract the models' functionality"""
import abc
from enum import Enum
from pytriton.decorators import batch


class AbstractModel(abc.ABC):
    """An abstraction of a generative AI model"""

    @abc.abstractmethod
    def load(self, model_path):
        """An abstract method for loading a model"""
        pass

    @abc.abstractmethod
    def bind(self, triton):
        pass

    @property
    @abc.abstractmethod
    def rank(self):
        pass

    @abc.abstractmethod
    @batch
    def infer(self, **inputs):
        pass

    @abc.abstractmethod
    def generate(self, inputs):
        pass


class Task(Enum):
    """Task enum"""
    GENERATE = 0
    GET_ACTIVATIONS = 1
    EDIT_ACTIVATIONS = 2
