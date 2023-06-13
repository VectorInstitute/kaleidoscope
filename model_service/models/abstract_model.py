"""A module to abstract the models' functionality"""
import abc

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

    @abc.abstractmethod
    def get_activations(self, request):
        pass

    @abc.abstractmethod
    def edit_activations(self, request):
        pass
