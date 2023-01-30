from celery import shared_task

from models import ModelInstance

@shared_task
def verify_model_instance_health():
    model_instances = ModelInstance.get_current_instances()
    for model_instance in model_instances:
        if not model_instance.is_healthy():
            model_instance.destroy()

@shared_task
def launch_model_instance(model_instance_id):
    model_instance = ModelInstance.get_by_id(model_instance_id)
    model_instance.launch()

@shared_task
def shutdown_model_instance(model_instance_id):
    model_instance = ModelInstance.get_by_id(model_instance_id)
    model_instance.shutdown()