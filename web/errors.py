"""Module for model error states"""
from __future__ import annotations
import typing
from typing import Optional

if typing.TYPE_CHECKING:
    from models import ModelInstanceState


class InvalidStateError(Exception):
    """Class to represent a invalid model state"""

    def __init__(self, state: ModelInstanceState, message: Optional[str] = None):
        self.state = state
        if message is None:
            message = f"Invalid operation for model instance state: {self.state.__class__}"
        super().__init__(message)
