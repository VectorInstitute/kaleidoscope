from gateway_service import celery

@celery.task()
def add_together(a, b):
    return a + b

