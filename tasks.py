from celery import shared_task

from resources.models.models import ModelInstance


@shared_task
def verify_model_instance_health():
    model_instances = ModelInstance.query.all()
    for model_instance in model_instances:
        if not model_instance.is_healthy():
            model_instance.destroy()
