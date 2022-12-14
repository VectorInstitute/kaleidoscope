#!/usr/bin/env python3

from flask import Flask
from flask_ldap3_login import LDAP3LoginManager
from flask_jwt_extended import JWTManager
from celery import Celery

from config import Config
from auth import auth
from db import db
from resources.home import home
from resources.models import models

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    ldap_manager = LDAP3LoginManager(app)  
    jwt = JWTManager(app)

    app.register_blueprint(auth)
    app.register_blueprint(home.home_bp)
    app.register_blueprint(models.models_bp, url_prefix='/models')

    db.init_app(app)
    with app.app_context():
        db.create_all()

    return app

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_BACKEND_URL'],
        broker=app.config['CELERY_BROKER_URL'],
        include=['tasks']
    )
    celery.conf.update(app.config)

    celery.conf.beat_schedule = {
        "verify_health": {
            "task": "tasks.verify_model_instance_health",
            "schedule": 10.0
        }
    }

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    app.celery = celery
    return celery

app = create_app()
celery = make_celery(app)

if __name__ == "__main__":
    app.run(host=Config.GATEWAY_HOST, port=Config.GATEWAY_PORT)
