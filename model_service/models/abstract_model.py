"""A module to abstract the models' functionality"""
import abc


class AbstractModel(abc.ABC):
    """An abstraction of a generative AI model"""

    @abc.abstractmethod
    def load(self, device, model_path):
        """An abstract method for loading a model"""

    @abc.abstractmethod
    def module_names(self):
        """An abstract method for getting module names"""

    @abc.abstractmethod
    def generate(self, request):
        """An abstract method for generating text"""

    @abc.abstractmethod
    def get_activations(self, request):
        pass

    @abc.abstractmethod
    def edit_activations(self, request):
        pass
