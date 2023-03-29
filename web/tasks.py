from datetime import datetime
from celery import shared_task

from models import ModelInstance
from config import Config

import sys
from flask import current_app

def verify_model_instance_timeout(model_instance, timeout):


    current_app.logger.info(model_instance.state_name)
    
    if not model_instance.state_name == 'ACTIVE':
        return False

    last_event_datetime = model_instance.updated_at

    last_generation = model_instance.last_generation()
    if last_generation:
        last_event_datetime = last_generation.created_at

    current_app.logger.info(last_event_datetime)
    current_app.logger.info(last_generation)
    current_app.logger.info((datetime.now() - last_event_datetime) > timeout)
    current_app.logger.info((datetime.now() - last_event_datetime))

    return (datetime.now() - last_event_datetime) > timeout

@shared_task
def verify_model_instance_health():
    current_model_instances = ModelInstance.find_current_instances()
    for model_instance in current_model_instances:
        if not model_instance.is_healthy() or verify_model_instance_timeout(model_instance, Config.MODEL_INSTANCE_TIMEOUT):
            model_instance.shutdown()


@shared_task
def launch_model_instance(model_instance_id):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.launch()


@shared_task
def shutdown_model_instance(model_instance_id):
    model_instance = ModelInstance.find_by_id(model_instance_id)
    model_instance.shutdown()
