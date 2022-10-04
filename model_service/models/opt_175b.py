import argparse
import logging
import numpy as np
import time
import torch

from .abstract_model import AbstractModel
from werkzeug.exceptions import HTTPException

class OPT_175B(AbstractModel):

    def __init__(self):
        self.model = None


    def load(self):
        # Add code to load the model here
        self.model = None
        print("Called OPT_175B.load()")


    def module_names(self):
        print("Called OPT_175B.module_names()")
        return "Placeholder return text for OPT_175B.module_names()"


    def generate_text(self, prompt, args):
        print("Called OPT_175B.generate_text()")
        return "Placeholder return text for OPT_175B.generate_text()"