from enum import Enum
import requests
from db import db, BaseMixin


class ModelInstanceStates(Enum):
    LAUNCHING = 0
    LOADING = 1
    ACTIVE = 2
    FAILED = 3
    EXPIRED = 4


class ModelInstance(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String)
    host = db.Column(db.String)
    state = db.Column(db.Enum(ModelInstanceStates), default=ModelInstanceStates.LAUNCHING)
    generations = db.relationship('ModelInstanceGeneration', backref='ModelInstance', lazy=True)

    def __init__(self, type, host):
        self.type = type
        self.host = host
        
    @property
    def base_addr(self):
        return f"http://{self.host}"

    def is_healthy(self):
        try:
            response = requests.get(self.base_addr + "/health")
            return response.status_code == 200
        except:
            return False


class ModelInstanceGeneration(BaseMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_instance_id = db.Column(db.Integer, db.ForeignKey("model_instance.id"))
    prompt = db.Column(db.String)

    def __init__(
        self,
        model_instance_id,
        prompt
    ):
        self.model_instance_id = model_instance_id
        self.prompt = prompt
