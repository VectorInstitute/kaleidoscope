from datetime import datetime
from celery import shared_task

from models import ModelInstance
from config import Config

from flask import current_app


@shared_task
def verify_model_instance_health():
    current_model_instances = ModelInstance.find_current_instances()
    for model_instance in current_model_instances:
        current_app.logger.info(model_instance.id)
        current_app.logger.info(model_instance.state_name)
        if not model_instance.is_healthy() or model_instance.is_timed_out(Config.MODEL_INSTANCE_TIMEOUT):
            model_instance.shutdown()

@shared_task
def launch_model_instance(model_instance_id):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.launch()

@shared_task
def shutdown_model_instance(model_instance_id):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.shutdown()
